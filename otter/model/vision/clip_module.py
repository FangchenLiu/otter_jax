from typing import Union, Optional, Tuple
import flax
from flax.linen import combine_masks, make_causal_mask
from jax import lax
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.attention import dot_product_attention_weights
from transformers import CLIPVisionConfig, CLIPTextConfig
from transformers.modeling_flax_outputs import ModelOutput, FlaxBaseModelOutput
from transformers.models.clip.modeling_flax_clip import (
    FlaxCLIPVisionEmbeddings,
    FlaxCLIPAttention,
    FlaxCLIPMLP,
    FlaxCLIPOutput,
    FlaxCLIPTextTransformer,
)
from transformers import CLIPTextConfig, CLIPVisionConfig, CLIPConfig
import jax


@flax.struct.dataclass
class FlaxBaseClearCLIPOutputWithPooling(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: jnp.ndarray = None
    pooler_output: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    post_ln_hidden_state: Optional[jnp.ndarray] = None


class FlaxCLIPEncoder(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = FlaxCLIPLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        inputs_embeds,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        cross_attention_features=None,
        return_last_self_attn: bool = False,
    ):
        return self.layers(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cross_attention_features=cross_attention_features,
            return_last_self_attn=return_last_self_attn,
        )


class FlaxClearCLIPModule(nn.Module):
    config: CLIPConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        text_config = self.config.text_config
        vision_config = self.config.vision_config

        self.projection_dim = self.config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = FlaxCLIPTextTransformer(text_config, dtype=self.dtype)
        self.vision_model = FlaxClearCLIPVisionTransformer(
            vision_config, dtype=self.dtype
        )

        self.visual_projection = nn.Dense(
            self.projection_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
        )
        self.text_projection = nn.Dense(
            self.projection_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
        )

        self.logit_scale = self.param(
            "logit_scale",
            lambda _, shape: jnp.ones(shape) * self.config.logit_scale_init_value,
            [],
        )

    def __call__(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / jnp.linalg.norm(
            image_embeds, axis=-1, keepdims=True
        )
        text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)

        # cosine similarity as logits
        logit_scale = jnp.exp(self.logit_scale)
        logits_per_text = jnp.matmul(text_embeds, image_embeds.T) * logit_scale
        logits_per_image = logits_per_text.T

        if not return_dict:
            return (
                logits_per_image,
                logits_per_text,
                text_embeds,
                image_embeds,
                text_outputs,
                vision_outputs,
            )

        return FlaxCLIPOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    def get_text_features(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train=False,
    ):

        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape
            )

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        input_ids = jnp.array(input_ids, dtype="i4")
        attention_mask = jnp.array(attention_mask, dtype="i4")
        position_ids = jnp.array(position_ids, dtype="i4")

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=not train,
        )
        last_hidden_state = text_outputs[0]
        pooled_output = text_outputs[1]
        pooled_output = self.text_model.final_layer_norm(pooled_output)
        text_features = self.text_projection(pooled_output)
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)
        last_hidden_state = self.text_projection(last_hidden_state)
        return last_hidden_state, text_features

    def get_image_features(
        self,
        pixel_values,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train=False,
    ):
        r"""
        Args:
            pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained
                using [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.

        Returns:
            image_features (`jnp.ndarray` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`FlaxCLIPVisionModel`]

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, FlaxCLIPModel

        >>> model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="np")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        pixel_values = jnp.array(pixel_values, dtype=jnp.float32)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            deterministic=not train,
            post_ln_hidden_state=True,
        )
        post_ln_state = vision_outputs[-1]  # post_ln_state
        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)
        post_ln_state = self.visual_projection(post_ln_state)
        return post_ln_state, image_features


class FlaxClearCLIPVisionTransformer(nn.Module):
    config: CLIPVisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embeddings = FlaxCLIPVisionEmbeddings(self.config, dtype=self.dtype)
        self.pre_layrnorm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )
        self.encoder = FlaxCLIPEncoder(self.config, dtype=self.dtype)
        self.post_layernorm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )

    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        cross_attention_features=None,
        post_ln_hidden_state: Optional[bool] = False,
        return_dict: bool = True,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cross_attention_features=cross_attention_features,
            return_last_self_attn=True,
        )

        post_ln_hidden_state = encoder_outputs[0]
        post_ln_hidden_state = self.post_layernorm(post_ln_hidden_state)
        last_hidden_state = encoder_outputs[0]
        # pooled_output = last_hidden_state[:, 0, :]
        # pooled_output = self.post_layernorm(pooled_output)
        pooled_output = post_ln_hidden_state[:, 0, :]
        post_ln_hidden_state = post_ln_hidden_state[
            :, 1:, :
        ]  # remove cls token from the hidden states

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return FlaxBaseClearCLIPOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            post_ln_hidden_state=post_ln_hidden_state,
        )


class FlaxCLIPLayerCollection(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            FlaxCLIPEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        cross_attention_features=None,
        return_last_self_attn: bool = False,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if cross_attention_features is not None:
                caf = cross_attention_features[idx]
            else:
                caf = None

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                cross_attention_feature=caf,
                deterministic=deterministic,
                output_attentions=output_attentions,
                return_self_attn=return_last_self_attn
                and (idx == len(self.layers) - 1),
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class FlaxCLIPAttentionFused(FlaxCLIPAttention):
    def __call__(
        self,
        hidden_states,
        cross_attention_feature=None,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        if cross_attention_feature is not None:
            # concate with hidden states
            kv_hidden_states = jnp.concatenate(
                [hidden_states, cross_attention_feature], axis=-2
            )
        else:
            kv_hidden_states = hidden_states
        query = self.q_proj(hidden_states)
        key = self.k_proj(kv_hidden_states)
        value = self.v_proj(kv_hidden_states)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        causal_attention_mask = None
        if self.causal:
            query_length, key_length = query.shape[1], key.shape[1]
            causal_attention_mask = self.causal_mask[
                :, :, key_length - query_length : key_length, :key_length
            ]

        if attention_mask is not None and causal_attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_mask = combine_masks(
                attention_mask, causal_attention_mask, dtype="i4"
            )
        elif causal_attention_mask is not None:
            attention_mask = causal_attention_mask
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        if attention_mask is not None:
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                    self.dtype
                ),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxCLIPEncoderLayer(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self_attn = FlaxCLIPAttentionFused(self.config, dtype=self.dtype)
        self.layer_norm1 = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )
        self.mlp = FlaxCLIPMLP(self.config, dtype=self.dtype)
        self.layer_norm2 = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )

    def __call__(
        self,
        hidden_states,
        attention_mask,
        cross_attention_feature=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        return_self_attn: bool = False,
    ):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            cross_attention_feature=cross_attention_feature,
        )
        hidden_states = attn_outputs[0]
        if return_self_attn:
            # print("returning self attn")
            outputs = (hidden_states,)
            if output_attentions:
                outputs += attn_outputs[1:]
            return outputs

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += attn_outputs[1:]

        return outputs
