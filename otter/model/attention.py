from typing import Callable, Optional
import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange, repeat


def mask_union(mask1, mask2):
    return jnp.logical_or(mask1 > 0, mask2 > 0).astype(jnp.float32)


def mask_intersection(mask1, mask2):
    return jnp.logical_and(mask1 > 0, mask2 > 0).astype(jnp.float32)


def mask_not(mask):
    return 1.0 - mask


def mask_select(mask, this, other=None):
    if other is None:
        other = jnp.array(0, dtype=this.dtype)
    if len(this.shape) == 3:
        mask = jnp.expand_dims(mask, axis=-1)
    return jnp.where(mask == 0.0, this, other)


def no_mask(x):
    return jnp.zeros(x.shape[:2])


def all_mask(x):
    return jnp.ones(x.shape[:2])


def patch_mse_loss(patch_output, patch_target, valid=None):
    if valid is None:
        valid = all_mask(patch_target)
    valid_ratio = jnp.sum(valid, axis=-1) / valid.shape[-1]
    return jnp.mean(
        jnp.mean(
            jnp.where(
                valid > 0.0,
                jnp.mean(jnp.square(patch_target - patch_output), axis=-1),
                jnp.array(0.0),
            ),
            axis=-1,
        )
        / valid_ratio
    )


def extract_patches(inputs, patch_size):
    batch, height, width, channels = inputs.shape
    height, width = height // patch_size, width // patch_size
    x = jnp.reshape(inputs, (batch, height, patch_size, width, patch_size, channels))
    x = jnp.swapaxes(x, 2, 3)
    x = jnp.reshape(x, (batch, height * width, patch_size**2 * channels))
    return x


def merge_patches(inputs, patch_size):
    batch, length, _ = inputs.shape
    height = width = int(length**0.5)
    x = jnp.reshape(inputs, (batch, height, width, patch_size, patch_size, -1))
    x = jnp.swapaxes(x, 2, 3)
    x = jnp.reshape(x, (batch, height * patch_size, width * patch_size, -1))
    return x


def extract_patches_video(
    inputs: jnp.ndarray, patch_size: int, time_size: int
) -> jnp.ndarray:
    batch, time, height, width, channels = inputs.shape
    time = time // time_size
    height = height // patch_size
    width = width // patch_size

    x = jnp.reshape(
        inputs,
        (batch, time, time_size, height, patch_size, width, patch_size, channels),
    )  # B(0), T(1), T_S(2), H(3), H_S(4), W(5), W_S(6), C(7)
    x = x.transpose(0, 1, 3, 5, 2, 4, 6, 7)
    x = jnp.reshape(
        x, (batch, time, height * width, time_size * patch_size**2 * channels)
    )
    return x


def merge_patches_video(
    inputs: jnp.ndarray, patch_size: int, time_size: int
) -> jnp.ndarray:
    batch, time, length, _ = inputs.shape
    height = width = int(length**0.5)
    x = jnp.reshape(
        inputs, (batch, time, height, width, time_size, patch_size, patch_size, -1)
    )
    x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)
    x = jnp.reshape(
        x, (batch, time * time_size, height * patch_size, width * patch_size, -1)
    )
    return x


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


def interpolate_positional_embedding(pos_embed, orig_length, new_length):
    assert pos_embed.shape[1] == orig_length
    D = pos_embed.shape[2]
    orig_grid = jnp.arange(orig_length, dtype=jnp.float32)
    new_grid = jnp.linspace(0, orig_length - 1, new_length)
    new_pos_embed = []
    for i in range(D):
        new_pos_embed.append(jnp.interp(new_grid, orig_grid, pos_embed[0, :, i]))

    new_pos_embed = jnp.stack(new_pos_embed, axis=-1)
    print("interpolate positional embedding", new_pos_embed.shape)
    new_pos_embed = jnp.expand_dims(new_pos_embed, 0)
    return new_pos_embed


def get_2d_sincos_pos_embed(embed_dim, length):
    grid_size = int(length**0.5)
    assert grid_size * grid_size == length, f"grid_size: {grid_size}, length: {length}"

    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0)


def index_sequence(x, ids):
    return x[:, ids, ...]


class MLP(nn.Module):
    hidden_dim: int
    output_dim: int
    depth: int

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i in range(self.depth):
            y = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(
                x
            )
            y = nn.gelu(y)
            y = nn.LayerNorm()(y)
            if i > 0:
                x = x + y
            else:
                x = y

        x = nn.Dense(self.output_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        return x


class TransformerMLP(nn.Module):
    dim: int = 256
    out_dim: int = 256
    dropout: float = 0.0
    kernel_init: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        x = nn.Dense(self.dim, kernel_init=self.kernel_init, name="fc1")(inputs)

        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic)
        x = nn.Dense(self.out_dim, kernel_init=self.kernel_init, name="fc2")(x)
        x = nn.Dropout(self.dropout)(x, deterministic)

        return x


class Attention(nn.Module):
    """Modified from flax_models to support mask"""

    dim: int
    num_heads: int = 8
    use_bias: bool = False
    att_drop: float = 0
    proj_drop: float = 0
    kernel_init: Callable = nn.initializers.xavier_uniform()
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None, attn_mask=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        batch, n, channels = inputs.shape
        scale = (self.dim // self.num_heads) ** -0.5
        qkv = nn.Dense(
            self.dim * 3,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
        )(inputs)
        qkv = jnp.reshape(
            qkv, (batch, n, 3, self.num_heads, channels // self.num_heads)
        )
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        # shape: (3, batch, num_heads, n, channels // num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q @ jnp.swapaxes(k, -2, -1)) * scale
        # shape of attention: (batch, num_heads, n, n)
        if attn_mask is not None:
            attention = jnp.where(attn_mask > 0, attention, jnp.array(-1e7))

        attention = nn.softmax(attention, axis=-1)
        self.sow("intermediates", "attention", attention)
        attention = nn.Dropout(self.att_drop)(attention, deterministic)

        output = (attention @ v).swapaxes(1, 2).reshape(batch, n, channels)
        x = nn.Dense(self.dim, kernel_init=nn.initializers.xavier_uniform())(output)

        x = nn.Dropout(self.proj_drop)(x, deterministic)

        return x


import equinox as eqx
from flax.linen.attention import (
    dot_product_attention,
    combine_masks,
    dot_product_attention_weights,
)
import functools
from typing import Any, Callable, Optional, Tuple, Union, overload
import warnings

from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import default_kernel_init
from flax.linen.linear import DenseGeneral
from flax.linen.linear import DotGeneralT
from flax.linen.linear import PrecisionLike
from flax.linen.module import compact
from flax.linen.module import merge_param
from flax.linen.module import Module
from flax.linen.normalization import LayerNorm
import jax
from jax import lax
from jax import random
import jax.numpy as jnp


PRNGKey = jax.Array
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


def get_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights.

    Note: query, key, value needn't have any batch dimensions.

    Args:
      query: queries for calculating attention with shape of `[batch..., q_length,
        num_heads, qk_depth_per_head]`.
      key: keys for calculating attention with shape of `[batch..., kv_length,
        num_heads, qk_depth_per_head]`.
      value: values to be used in attention with shape of `[batch..., kv_length,
        num_heads, v_depth_per_head]`.
      bias: bias for the attention weights. This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
        incorporating causal masks, padding masks, proximity bias, etc.
      mask: mask for the attention weights. This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
        incorporating causal masks. Attention weights are masked out if their
        corresponding mask value is `False`.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      dtype: the dtype of the computation (default: infer from inputs)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.

    Returns:
      Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
    """
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), "q, k, v batch dims must match."
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    # compute attention weights
    attn_weights = dot_product_attention_weights(
        query,
        key,
        bias,
        mask,
        broadcast_dropout,
        dropout_rng,
        dropout_rate,
        deterministic,
        dtype,
        precision,
    )

    # return weighted sum over values for each query position
    return attn_weights


class MultiHeadDotProductAttention(eqx.Module):
    """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: infer from inputs and params)
      param_dtype: the dtype passed to parameter initializers (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts query,
        key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
        num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
      normalize_qk: should QK normalization be applied (arxiv.org/abs/2302.05442).
    """

    max_seq_length: int
    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    use_bias: bool = True
    attention_fn: Callable[..., Array] = dot_product_attention
    decode: bool = False
    normalize_qk: bool = False
    # Deprecated, will be removed.
    qkv_dot_general: Optional[DotGeneralT] = None
    out_dot_general: Optional[DotGeneralT] = None
    qkv_dot_general_cls: Any = None
    out_dot_general_cls: Any = None

    def setup(self):

        self.kv_cache_index = eqx.nn.StateIndex(
            (
                jnp.zeros(shape=(self.max_seq_len, self.num_heads, self.qkv_features)),
                jnp.zeros(shape=(self.max_seq_len, self.num_heads, self.qkv_features)),
                jnp.zeros(shape=(1,), dtype=jnp.int32),
            )
        )

    @overload
    def __call__(
        self,
        inputs_q: Array,
        inputs_k: Optional[Array] = None,
        inputs_v: Optional[Array] = None,
        *,
        mask: Optional[Array] = None,
        deterministic: Optional[bool] = None,
        dropout_rng: Optional[PRNGKey] = None,
    ): ...

    @overload
    def __call__(
        self,
        inputs_q: Array,
        *,
        inputs_kv: Array = None,
        mask: Optional[Array] = None,
        deterministic: Optional[bool] = None,
        dropout_rng: Optional[PRNGKey] = None,
    ): ...

    @compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_k: Optional[Array] = None,
        inputs_v: Optional[Array] = None,
        *,
        inputs_kv: Optional[Array] = None,
        mask: Optional[Array] = None,
        deterministic: Optional[bool] = None,
        dropout_rng: Optional[PRNGKey] = None,
    ):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        If both inputs_k and inputs_v are None, they will both copy the value of
        inputs_q (self attention).
        If only inputs_v is None, it will copy the value of inputs_k.

        Args:
          inputs_q: input queries of shape `[batch_sizes..., length, features]`.
          inputs_k: key of shape `[batch_sizes..., length, features]`. If None,
            inputs_k will copy the value of inputs_q.
          inputs_v: values of shape `[batch_sizes..., length, features]`. If None,
            inputs_v will copy the value of inputs_k.
          inputs_kv: key/values of shape `[batch_sizes..., length, features]`. If
            None, inputs_kv will copy the value of inputs_q. This arg will be
            deprecated soon. Use inputs_k and inputs_v instead.
          mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
            key/value_length]`. Attention weights are masked out if their
            corresponding mask value is `False`.
          deterministic: if false, the attention weight is masked randomly using
            dropout, whereas if true, the attention weights are deterministic.
          dropout_rng: optional rng key to pass to the attention layer's dropout
            mask. Otherwise, self.make_rng('dropout') is used instead.

        Returns:
          output of shape `[batch_sizes..., length, features]`.
        """
        if inputs_kv is not None:
            if inputs_k is not None or inputs_v is not None:
                raise ValueError(
                    "If either `inputs_k` or `inputs_v` is not None, "
                    "`inputs_kv` must be None. If `inputs_kv` is not None, both `inputs_k` "
                    "and `inputs_v` must be None. We recommend using `inputs_k` and "
                    "`inputs_v` args, since `inputs_kv` will be deprecated soon. See "
                    "https://github.com/google/flax/discussions/3389 for more "
                    "information."
                )
            inputs_k = inputs_v = inputs_kv
            warnings.warn(
                "The inputs_kv arg will be deprecated soon. "
                "Use inputs_k and inputs_v instead. See "
                "https://github.com/google/flax/discussions/3389 "
                "for more information.",
                DeprecationWarning,
            )
        else:
            if inputs_k is None:
                if inputs_v is not None:
                    raise ValueError(
                        "`inputs_k` cannot be None if `inputs_v` is not None. "
                        "To have both `inputs_k` and `inputs_v` be the same value, pass in the "
                        "value to `inputs_k` and leave `inputs_v` as None."
                    )
                inputs_k = inputs_q
            if inputs_v is None:
                inputs_v = inputs_k
            elif inputs_v.shape[-1] == inputs_v.shape[-2]:
                warnings.warn(
                    f"You are passing an array of shape {inputs_v.shape} "
                    "to the `inputs_v` arg, when you may have intended "
                    "to pass it to the `mask` arg. As of Flax version "
                    "0.7.4, the function signature of "
                    "MultiHeadDotProductAttention's `__call__` method "
                    "has changed to `__call__(inputs_q, inputs_k=None, "
                    "inputs_v=None, *, inputs_kv=None, mask=None, "
                    "deterministic=None)`. Use the kwarg `mask` instead. "
                    "See https://github.com/google/flax/discussions/3389 "
                    "and read the docstring for more information.",
                    DeprecationWarning,
                )

        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, (
            f"Memory dimension ({qkv_features}) must be divisible by number of"
            f" heads ({self.num_heads})."
        )
        head_dim = qkv_features // self.num_heads

        dense = functools.partial(
            DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            dot_general=self.qkv_dot_general,
            dot_general_cls=self.qkv_dot_general_cls,
        )
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        query, key, value = (
            dense(name="query")(inputs_q),
            dense(name="key")(inputs_k),
            dense(name="value")(inputs_v),
        )

        if self.normalize_qk:
            # Normalizing query and key projections stabilizes training with higher
            # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
            query = LayerNorm(name="query_ln", use_bias=False)(query)  # type: ignore[call-arg]
            key = LayerNorm(name="key_ln", use_bias=False)(key)  # type: ignore[call-arg]

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.decode:
            # detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable("cache", "cached_key")
            cached_key = self.variable(
                "cache", "cached_key", jnp.zeros, key.shape, key.dtype
            )
            cached_value = self.variable(
                "cache", "cached_value", jnp.zeros, value.shape, value.dtype
            )
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
            )
            if is_initialized:
                (
                    *batch_dims,
                    max_length,
                    num_heads,
                    depth_per_head,
                ) = cached_key.value.shape
                # shape check of cached keys against query input
                expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
                if expected_shape != query.shape:
                    raise ValueError(
                        "Autoregressive cache shape error, "
                        "expected query shape %s instead got %s."
                        % (expected_shape, query.shape)
                    )
                # update key, value caches with our new 1d spatial slices
                cur_index = cache_index.value
                indices: tuple[Union[int, jax.Array], ...] = (0,) * len(batch_dims) + (
                    cur_index,
                    0,
                    0,
                )
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(cached_value.value, value, indices)
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1
                # causal mask for cached decoder self-attention:
                # our single query position should only attend to those key
                # positions that have already been generated and cached,
                # not the remaining zero elements.
                mask = combine_masks(
                    mask,
                    jnp.broadcast_to(
                        jnp.arange(max_length) <= cur_index,
                        tuple(batch_dims) + (1, 1, max_length),
                    ),
                )

        if self.dropout_rate > 0.0:  # Require `deterministic` only if using dropout.
            m_deterministic = merge_param(
                "deterministic", self.deterministic, deterministic
            )
            if not m_deterministic and dropout_rng is None:
                dropout_rng = self.make_rng("dropout")
        else:
            m_deterministic = True

        # apply attention
        # x = self.attention_fn(
        #     query,
        #     key,
        #     value,
        #     mask=mask,
        #     dropout_rng=dropout_rng,
        #     dropout_rate=self.dropout_rate,
        #     broadcast_dropout=self.broadcast_dropout,
        #     deterministic=m_deterministic,
        #     dtype=self.dtype,
        #     precision=self.precision,
        # )  # pytype: disable=wrong-keyword-args

        attn_weights = get_attention(
            query,
            key,
            value,
            mask=mask,
            broadcast_dropout=self.broadcast_dropout,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            deterministic=m_deterministic,
            dtype=self.dtype,
            precision=self.precision,
        )

        self.sow("intermediates", "attn_weights", attn_weights)

        x = jnp.einsum(
            "...hqk,...khd->...qhd", attn_weights, value, precision=self.precision
        )

        # back to the original inputs dimensions
        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            dot_general=self.out_dot_general,
            dot_general_cls=self.out_dot_general_cls,
            name="out",  # type: ignore[call-arg]
        )(x)
        return out
