# adapted from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
from typing import Callable, Optional

import flax.linen as nn
import jax.numpy as jnp

from otter.typing import Array, Dtype, PRNGKey, Shape
from otter.model.attention import Attention


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
        ),
        0,
    )


class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs.

    Attributes:
      posemb_init: positional embedding initializer.
    """

    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

    @nn.compact
    def __call__(self, inputs):
        """Applies the AddPositionEmbs module.

        Args:
          inputs: Inputs to the layer.

        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, (
            "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        )
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param("pos_embedding", self.posemb_init, pos_emb_shape)
        return inputs + pe


class AddSinusoidPositionEmbs(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        """Applies the AddPositionEmbs module.

        Args:
          inputs: Inputs to the layer.

        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, (
            "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        )
        pe = get_1d_sincos_pos_embed(inputs.shape[-1], inputs.shape[1])
        return inputs + pe


class GetSinusoidPositionEmbs(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        """Get the AddPositionEmbs module.

        Args:
          inputs: Inputs to the layer.

        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, (
            "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        )
        pe = get_1d_sincos_pos_embed(inputs.shape[-1], inputs.shape[1])
        return pe


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
        nn.initializers.xavier_uniform()
    )
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(
        stddev=1e-6
    )

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(inputs)

        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)

        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
        return output


class CrossAttentionPooling(nn.Module):

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 8
    num_readouts: int = 1
    num_layers: int = 1
    use_query_tokens: bool = False
    add_pe: bool = True
    dropout_rate: float = (0.0,)
    out_dim: Optional[int] = (None,)

    @nn.compact
    def __call__(self, kv, query=None, train=True):
        batch_size, l, d = kv.shape

        if query is None:
            assert self.use_query_tokens, "use learnable query tokens"
            query = self.param(
                "query",
                nn.initializers.xavier_uniform(),
                (1, self.num_readouts, self.out_dim),
                kv.dtype,
            )
            query = jnp.tile(query, [batch_size, 1, 1])
        else:
            assert not self.use_query_tokens, "use parsed-in query tokens"

        if self.add_pe:
            query = AddSinusoidPositionEmbs(name="posembed_query")(query)
            kv = AddSinusoidPositionEmbs(name="posembed_kv")(kv)

        for _ in range(self.num_layers):
            query = CrossAttnBlock(
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                out_dim=self.out_dim,
            )(query, kv, deterministic=not train)

        out = nn.LayerNorm()(query)
        out = out.reshape(batch_size, -1, query.shape[-1])
        return out


class CrossAttnBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """

    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, query, kv, deterministic):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert query.ndim == 3, f"Expected (batch, seq, hidden) got {query.shape}"
        x = nn.LayerNorm(dtype=self.dtype)(query)

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            out_features=self.out_dim,
        )(x, kv, mask=None)

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + query

        # MLP block.
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
            out_dim=self.out_dim,
        )(y, deterministic=deterministic)

        return x + y


class CrossAttentionPoolingNoResidual(nn.Module):

    num_heads: int = 8
    num_readouts: int = 1
    num_layers: int = 1
    use_query_tokens: bool = False
    dropout_rate: float = (0.0,)
    out_dim: Optional[int] = (None,)

    @nn.compact
    def __call__(self, kv, query=None, train=True):
        batch_size, l, d = kv.shape

        if query is None:
            assert self.use_query_tokens, "use learnable query tokens"
            query = self.param(
                "query",
                nn.initializers.xavier_uniform(),
                (1, self.num_readouts, self.out_dim),
                kv.dtype,
            )
            query = jnp.tile(query, [batch_size, 1, 1])
        else:
            assert not self.use_query_tokens, "use parsed-in query tokens"

        for _ in range(self.num_layers):
            query = CrossAttnBlockNoResidual(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                out_dim=self.out_dim,
            )(query, kv, deterministic=not train)

        out = nn.LayerNorm()(query)
        out = out.reshape(batch_size, -1, query.shape[-1])
        return out


class CrossAttnBlockNoResidual(nn.Module):
    """Transformer encoder layer.

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """

    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, query, kv, deterministic):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert query.ndim == 3, f"Expected (batch, seq, hidden) got {query.shape}"
        x = nn.LayerNorm(dtype=self.dtype)(query)

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            out_features=self.out_dim,
        )(x, kv, mask=None)

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        return x


class TextImageCrossAttention(nn.Module):
    """Transformer encoder layer.
    This is a special attention block because there is no w_q, w_k, w_v

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """

    init_temperature: float = (
        0.005  # need to make this sufficient small to extract good features
    )
    use_learnable_temperature: bool = False
    add_pe: bool = False
    get_pe: bool = False

    def setup(self) -> None:
        if self.use_learnable_temperature:
            self.temperature = self.param(
                "temperature",
                lambda _, shape: jnp.ones(shape) * jnp.log(1 / self.init_temperature),
                [],
            )
        else:
            self.temperature = jnp.log(1 / self.init_temperature)

    @nn.compact
    def __call__(self, text, image):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert text.ndim == 3, f"Expected (batch, seq, hidden) got {text.shape}"
        assert (
            text.shape[-1] == image.shape[-1]
        ), f"Expected text and image to have the same hidden dimension, got {text.shape} and {image.shape}"

        # getpe and addpe can't be true at the same time
        assert not (self.get_pe and self.add_pe)

        d_k = image.shape[-1]
        x = jnp.einsum("btf,bif->bti", text, image)
        # self.sow("intermediates", 'text', text)
        # self.sow("intermediates", 'image', image)

        self.sow("intermediates", "text_img_dot_bt", x)

        temperature = jnp.clip(self.temperature, 0, 4.6052)
        temperature = jnp.exp(temperature)

        x = x * temperature  # b, 32, 196

        self.sow("intermediates", "text_img_dot", x)
        x = nn.softmax(x, axis=-1)
        # self.sow("intermediates", 'text_img_softmax', x)

        # add_pe and get_pe can't be true at the same time
        if self.get_pe:
            image = GetSinusoidPositionEmbs(name="image")(image)
        elif self.add_pe:
            image = AddSinusoidPositionEmbs(name="image")(image)

        x = jnp.einsum("bti,bif->btf", x, image)
        return x


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """

    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, attention_mask, *, deterministic):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads,
        )(x, x, mask=attention_mask)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
        )(y, deterministic=deterministic)

        return x + y


class Transformer(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate in self attention.
    """

    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0

    @nn.compact
    def __call__(self, x, train, attention_mask=None):
        """Applies Transformer model on the inputs.

        Args:
          x: Inputs to the layer.
          train: Set to `True` when training.

        Returns:
          output of a transformer encoder.
        """
        x = AddSinusoidPositionEmbs(name="posembed_input")(x)
        assert x.ndim == 3  # (batch, len, emb)
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f"encoderblock_{lyr}",
                num_heads=self.num_heads,
            )(x, attention_mask, deterministic=not train)
        encoded = nn.LayerNorm(name="encoder_norm")(x)

        return encoded
