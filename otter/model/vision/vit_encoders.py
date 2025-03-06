import functools as ft
from typing import Callable, Sequence, TypeVar

from flax import linen as nn
import jax.numpy as jnp

from otter.model.vision.film_conditioning_layer import FilmConditioning
from otter.model.transformer import AddSinusoidPositionEmbs, Encoder1DBlock
from transformers import FlaxViTModel, ViTConfig

T = TypeVar("T")


def weight_standardize(w, axis, eps):
    """Subtracts mean and divides by standard deviation."""
    w = w - jnp.mean(w, axis=axis)
    w = w / (jnp.std(w, axis=axis) + eps)
    return w


def normalize_images(img, img_norm_type="default"):
    if img_norm_type == "default":
        # put pixels in [-1, 1]
        return img.astype(jnp.float32) / 127.5 - 1.0
    elif img_norm_type == "imagenet":
        # put pixels in [0,1]
        img = img.astype(jnp.float32) / 255
        assert img.shape[-1] % 3 == 0, "images should have rgb channels!"

        # define pixel-wise mean/std stats calculated from ImageNet
        mean = jnp.array([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3))
        std = jnp.array([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3))

        # tile mean and std (to account for stacked early_fusion images)
        num_tile = (1, 1, 1, int(img.shape[-1] / 3))
        mean_tile = jnp.tile(mean, num_tile)
        std_tile = jnp.tile(std, num_tile)

        # tile the mean/std, normalize image, and return
        return (img - mean_tile) / std_tile

    elif img_norm_type == "clip":
        # put pixels in [0,1]
        img = img.astype(jnp.float32) / 255
        assert img.shape[-1] % 3 == 0, "images should have rgb channels!"

        # define pixel-wise mean/std stats calculated from ImageNet
        mean = jnp.array([0.48145466, 0.4578275, 0.40821073]).reshape((1, 1, 1, 3))
        std = jnp.array([0.26862954, 0.26130258, 0.27577711]).reshape((1, 1, 1, 3))

        # tile mean and std (to account for stacked early_fusion images)
        num_tile = (1, 1, 1, int(img.shape[-1] / 3))
        mean_tile = jnp.tile(mean, num_tile)
        std_tile = jnp.tile(std, num_tile)

        # tile the mean/std, normalize image, and return
        return (img - mean_tile) / std_tile

    else:
        raise ValueError(f"Unknown image normalization type: {img_norm_type}")


class StdConv(nn.Conv):
    """Convolution with weight standardization."""

    def param(self, name: str, init_fn: Callable[..., T], *init_args) -> T:
        param = super().param(name, init_fn, *init_args)
        if name == "kernel":
            param = weight_standardize(param, axis=[0, 1, 2], eps=1e-5)
        return param


class PatchEncoder(nn.Module):
    """Takes an image and breaks it up into patches of size (patch_size x patch_size),
    applying a fully connected network to each patch individually.

    The default "encoder" used by most ViTs in practice.
    """

    use_film: bool = False
    patch_size: int = 32
    num_features: int = 512
    img_norm_type: str = "imagenet"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True, cond_var=None):
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None
        assert (
            expecting_cond_var == received_cond_var
        ), "Only pass in cond var iff model expecting cond var"
        x = normalize_images(observations, self.img_norm_type)
        x = nn.Conv(
            features=self.num_features,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="embedding",
        )(x)
        if self.use_film:
            assert cond_var is not None, "Cond var is None, nothing to condition on"
            x = FilmConditioning()(x, cond_var)
        return x


class SmallStem(nn.Module):
    """Passes the image through a few light-weight convolutional layers,
    before patchifying the image. Empirically useful for many computer vision tasks.

    See Xiao et al: Early Convolutions Help Transformers See Better
    """

    use_film: bool = False
    patch_size: int = 32
    kernel_sizes: tuple = (3, 3, 3, 3)
    strides: tuple = (2, 2, 2, 2)
    features: tuple = (32, 96, 192, 384)
    padding: tuple = (1, 1, 1, 1)
    num_features: int = 512
    img_norm_type: str = "imagenet"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True, cond_var=None):
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None
        assert (
            expecting_cond_var == received_cond_var
        ), "Only pass in cond var iff model expecting cond var"

        x = normalize_images(observations, self.img_norm_type)
        for n, (kernel_size, stride, features, padding) in enumerate(
            zip(
                self.kernel_sizes,
                self.strides,
                self.features,
                self.padding,
            )
        ):
            x = StdConv(
                features=features,
                kernel_size=(kernel_size, kernel_size),
                strides=(stride, stride),
                padding=padding,
            )(x)
            x = nn.GroupNorm()(x)
            x = nn.relu(x)

        x = nn.Conv(
            features=self.num_features,
            kernel_size=(self.patch_size // 16, self.patch_size // 16),
            strides=(self.patch_size // 16, self.patch_size // 16),
            padding="VALID",
            name="embedding",
        )(x)
        if self.use_film:
            assert cond_var is not None, "Cond var is None, nothing to condition on"
            x = FilmConditioning()(x, cond_var)

        n, h, w, c = x.shape
        x = jnp.reshape(x, (n, h * w, c))
        return x


class ResidualUnit(nn.Module):
    """Bottleneck ResNet block."""

    features: int
    strides: Sequence[int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        needs_projection = x.shape[-1] != self.features * 4 or self.strides != (1, 1)

        residual = x
        if needs_projection:
            residual = StdConv(
                features=self.features * 4,
                kernel_size=(1, 1),
                strides=self.strides,
                use_bias=False,
                name="conv_proj",
            )(residual)
            residual = nn.GroupNorm(name="gn_proj")(residual)

        y = StdConv(
            features=self.features, kernel_size=(1, 1), use_bias=False, name="conv1"
        )(x)
        y = nn.GroupNorm(name="gn1")(y)
        y = nn.relu(y)
        y = StdConv(
            features=self.features,
            kernel_size=(3, 3),
            strides=self.strides,
            use_bias=False,
            name="conv2",
        )(y)
        y = nn.GroupNorm(name="gn2")(y)
        y = nn.relu(y)
        y = StdConv(
            features=self.features * 4, kernel_size=(1, 1), use_bias=False, name="conv3"
        )(y)

        y = nn.GroupNorm(name="gn3", scale_init=nn.initializers.zeros)(y)
        y = nn.relu(residual + y)
        return y


class ResNetStage(nn.Module):
    """A ResNet stage."""

    block_size: Sequence[int]
    nout: int
    first_stride: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = ResidualUnit(self.nout, strides=self.first_stride, name="unit1")(x)
        for i in range(1, self.block_size):
            x = ResidualUnit(self.nout, strides=(1, 1), name=f"unit{i + 1}")(x)
        return x


class ViTEncoder(nn.Module):
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
    dropout_rate: float = 0
    attention_dropout_rate: float = 0
    add_position_embedding: bool = True

    @nn.compact
    def __call__(self, x, *, train):
        """Applies Transformer model on the inputs.

        Args:
          x: Inputs to the layer.
          train: Set to `True` when training.

        Returns:
          output of a transformer encoder.
        """
        assert x.ndim == 3  # (batch, len, emb)

        if self.add_position_embedding:
            x = AddSinusoidPositionEmbs(name="posembed_input")(x)

        # Input Encoder
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f"encoderblock_{lyr}",
                num_heads=self.num_heads,
            )(x, deterministic=not train, attention_mask=None)
        encoded = nn.LayerNorm(name="encoder_norm")(x)

        return encoded


class ConvNet(nn.Module):
    """Resnet-v2 architecture used in the original ViT paper for hybrid (Resnet+ViT) architectures

    Mostly copied from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py

    There exist pre-trained parameters here: github.com/google-research/vision_transformer/
    """

    use_film: bool = False
    width: int = 1
    num_layers: tuple = tuple()
    img_norm_type: str = "imagenet"
    num_features: int = 512
    patch_size: int = 16

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True, cond_var=None):
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None
        assert (
            expecting_cond_var == received_cond_var
        ), "Only pass in cond var iff model expecting cond var"

        x = normalize_images(observations, self.img_norm_type)
        width = int(64 * self.width)
        x = StdConv(
            features=width,
            kernel_size=(7, 7),
            strides=(2, 2),
            use_bias=False,
            name="conv_root",
        )(x)
        x = nn.GroupNorm(name="gn_root")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        if self.num_layers:
            x = ResNetStage(
                block_size=self.num_layers[0],
                nout=width,
                first_stride=(1, 1),
                name="block1",
            )(x)
            for i, block_size in enumerate(self.num_layers[1:], 1):
                x = ResNetStage(
                    block_size=block_size,
                    nout=width * 2**i,
                    first_stride=(2, 2),
                    name=f"block{i + 1}",
                )(x)
                if self.use_film:
                    assert (
                        cond_var is not None
                    ), "Cond var is None, nothing to condition on"
                    x = FilmConditioning()(x, cond_var)

        x = nn.Conv(
            features=self.num_features,
            kernel_size=(
                self.patch_size // self.patch_size,
                self.patch_size // self.patch_size,
            ),
            strides=(
                self.patch_size // self.patch_size,
                self.patch_size // self.patch_size,
            ),
            padding="VALID",
            name="embedding",
        )(x)

        if self.use_film:
            assert cond_var is not None, "Cond var is None, nothing to condition on"
            x = FilmConditioning()(x, cond_var)

        n, h, w, c = x.shape
        x = jnp.reshape(x, (n, h * w, c))
        x = AddSinusoidPositionEmbs(name="posembed_input")(x)
        return x


class VisionTransformer(nn.Module):
    """Resnet-v2 architecture used in the original ViT paper for hybrid (Resnet+ViT) architectures

    Mostly copied from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py

    There exist pre-trained parameters here: github.com/google-research/vision_transformer/
    """

    use_film: bool = False
    width: int = 1
    num_layers: tuple = tuple()
    img_norm_type: str = "imagenet"
    num_features: int = 512
    patch_size: int = 16
    vit_encoder: nn.Module = ViTEncoder
    num_encoder_layers: int = 4
    encoder_mlp_dim: int = 1024
    encoder_num_heads: int = 8

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True, cond_var=None):
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None
        assert (
            expecting_cond_var == received_cond_var
        ), "Only pass in cond var iff model expecting cond var"

        x = normalize_images(observations, self.img_norm_type)
        width = int(64 * self.width)
        x = StdConv(
            features=width,
            kernel_size=(7, 7),
            strides=(2, 2),
            use_bias=False,
            name="conv_root",
        )(x)
        x = nn.GroupNorm(name="gn_root")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        if self.num_layers:
            x = ResNetStage(
                block_size=self.num_layers[0],
                nout=width,
                first_stride=(1, 1),
                name="block1",
            )(x)
            for i, block_size in enumerate(self.num_layers[1:], 1):
                x = ResNetStage(
                    block_size=block_size,
                    nout=width * 2**i,
                    first_stride=(2, 2),
                    name=f"block{i + 1}",
                )(x)
                if self.use_film:
                    assert (
                        cond_var is not None
                    ), "Cond var is None, nothing to condition on"
                    x = FilmConditioning()(x, cond_var)

        x = nn.Conv(
            features=self.num_features,
            kernel_size=(
                self.patch_size // self.patch_size,
                self.patch_size // self.patch_size,
            ),
            strides=(
                self.patch_size // self.patch_size,
                self.patch_size // self.patch_size,
            ),
            padding="VALID",
            name="embedding",
        )(x)

        if self.use_film:
            assert cond_var is not None, "Cond var is None, nothing to condition on"
            x = FilmConditioning()(x, cond_var)

        n, h, w, c = x.shape
        x = jnp.reshape(x, (n, h * w, c))
        x = self.vit_encoder(
            name="transformer",
            num_layers=self.num_encoder_layers,
            mlp_dim=self.encoder_mlp_dim,
            num_heads=self.encoder_num_heads,
        )(x, train=train)
        return x


class DinoViT16(nn.Module):
    image_norm_type: str = "imagenet"

    def setup(self):
        self.vision_transformer = FlaxViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True):
        x = normalize_images(observations, self.image_norm_type)
        x = self.vision_transformer(
            jnp.transpose(x, (0, 3, 1, 2)),
            train=train,
        ).last_hidden_state
        return x


class SmallStem16(SmallStem):
    patch_size: int = 16


class SmallStem32(SmallStem):
    patch_size: int = 32


class DinoViT16(VisionTransformer):
    """Vision Transformer with DINO pre-trainin"""

    img_norm_type: str = "imagenet"

    def setup(self):
        self.robot_vision_transformer = FlaxViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

    def __call__(self, observations: jnp.ndarray, train: bool = True):
        pixel_values = normalize_images(observations, self.img_norm_type)

        robot_vision_output = self.robot_vision_transformer(
            jnp.transpose(pixel_values, (0, 3, 1, 2)),
            output_hidden_states=True,
            train=train,
        )
        robot_tokens = robot_vision_output.last_hidden_state

        return robot_tokens


vit_encoder_configs = {
    "patchify-32-film": ft.partial(
        PatchEncoder,
        use_film=True,
        patch_size=32,
    ),
    "patchify-16-film": ft.partial(
        PatchEncoder,
        use_film=True,
        patch_size=16,
    ),
    "small-stem-8-film": ft.partial(
        SmallStem,
        use_film=True,
        patch_size=16,
        kernel_sizes=(3, 3, 3),
        strides=(2, 2, 2),
        features=(32, 96, 192),
        padding=(1, 1, 1),
    ),
    "small-stem-16": ft.partial(
        SmallStem,
        patch_size=16,
    ),
    "small-stem-16-film": ft.partial(
        SmallStem,
        patch_size=16,
        use_film=True,
    ),
    "small-vit-16": ft.partial(
        VisionTransformer,
        patch_size=16,
        num_layers=(2, 2, 2),
        num_encoder_layers=4,
        encoder_mlp_dim=768,
        encoder_num_heads=8,
    ),
    "small-conv-16": ft.partial(
        ConvNet,
        patch_size=16,
        num_layers=(2, 2, 2),
    ),
    "tiny-vit-14": ft.partial(
        VisionTransformer,
        patch_size=14,
        num_layers=(2, 2, 2),
        num_encoder_layers=2,
        encoder_mlp_dim=512,
        encoder_num_heads=8,
    ),
    "small-vit-14": ft.partial(
        VisionTransformer,
        patch_size=14,
        num_layers=(2, 2, 2),
        num_encoder_layers=4,
        encoder_mlp_dim=768,
        encoder_num_heads=8,
    ),
    "dino-vit-16": DinoViT16,
}
