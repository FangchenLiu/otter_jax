import flax.linen as nn
import jax
from flax.core.frozen_dict import freeze
from transformers import (
    CLIPTextConfig,
    CLIPVisionConfig,
    FlaxCLIPModel,
    FlaxCLIPPreTrainedModel,
    CLIPConfig,
)
from transformers.models.clip.modeling_flax_clip import (
    FlaxCLIPModule,
    CLIP_START_DOCSTRING,
    add_start_docstrings,
)
from otter.model.vision.clip_module import FlaxClearCLIPModule
from otter.model.vision.vit_encoders import normalize_images
import jax.numpy as jnp
from typing import Any, Optional, Tuple, Union


class AllTokenCLIPPreTrainedModel(FlaxCLIPPreTrainedModel):
    config_class = CLIPConfig
    module_class: nn.Module = None

    def __init__(
        self,
        config: CLIPConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (
                (1, 1),
                (
                    1,
                    config.vision_config.image_size,
                    config.vision_config.image_size,
                    3,
                ),
            )
        super().__init__(
            config, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init
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
        r"""
        Args:
            input_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)

        Returns:
            text_features (`jnp.ndarray` of shape `(batch_size, output_dim`): The text embeddings obtained by applying
            the projection layer to the pooled output of [`FlaxCLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, FlaxCLIPModel

        >>> model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="np")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape
            )

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _get_features(
            module, input_ids, attention_mask, position_ids, deterministic
        ):
            text_outputs = module.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
            )
            last_hidden_state = text_outputs[0]
            pooled_output = text_outputs[1]
            text_features = module.text_projection(pooled_output)
            return last_hidden_state, text_features

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            method=_get_features,
            rngs=rngs,
        )

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

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _get_features(module, pixel_values, deterministic):
            vision_outputs = module.vision_model(
                pixel_values=pixel_values, deterministic=deterministic
            )
            last_hidden_state = vision_outputs[0]
            pooled_output = vision_outputs[1]  # pooled_output
            image_features = module.visual_projection(pooled_output)
            return last_hidden_state, image_features

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            method=_get_features,
            rngs=rngs,
        )


@add_start_docstrings(CLIP_START_DOCSTRING)
class AllTokenFlaxCLIPModel(AllTokenCLIPPreTrainedModel):
    module_class = FlaxCLIPModule


class AllTokenClearCLIPPreTrainedModel(FlaxCLIPPreTrainedModel):
    config_class = CLIPConfig
    module_class: nn.Module = FlaxClearCLIPModule

    def __init__(
        self,
        config: CLIPConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (
                (1, 1),
                (
                    1,
                    config.vision_config.image_size,
                    config.vision_config.image_size,
                    3,
                ),
            )
        super().__init__(
            config=config,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
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

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _get_features(
            module, input_ids, attention_mask, position_ids, deterministic
        ):
            text_outputs = module.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
            )
            last_hidden_state = text_outputs[0]
            pooled_output = text_outputs[1]
            pooled_output = module.text_model.final_layer_norm(pooled_output)
            text_features = module.text_projection(pooled_output)
            last_hidden_state = module.text_model.final_layer_norm(last_hidden_state)
            last_hidden_state = module.text_projection(last_hidden_state)
            return last_hidden_state, text_features

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            method=_get_features,
            rngs=rngs,
        )

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

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _get_features(module, pixel_values, deterministic):
            # vision_outputs = module.vision_model(pixel_values=pixel_values, deterministic=deterministic)
            vision_outputs = module.vision_model(
                pixel_values=pixel_values,
                deterministic=deterministic,
                post_ln_hidden_state=True,
            )
            post_ln_state = vision_outputs[-1]  # post_ln_state
            pooled_output = vision_outputs[1]  # pooled_output
            image_features = module.visual_projection(pooled_output)
            post_ln_state = module.visual_projection(post_ln_state)
            return post_ln_state, image_features

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            method=_get_features,
            rngs=rngs,
        )


@add_start_docstrings(CLIP_START_DOCSTRING)
class AllTokenFlaxClearCLIPModel(AllTokenClearCLIPPreTrainedModel):
    module_class = FlaxClearCLIPModule
