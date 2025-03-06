import flax.linen as nn
import jax
from flax.core.frozen_dict import freeze
from otter.model.vision.clip_module import (
    FlaxClearCLIPVisionTransformer,
    FlaxClearCLIPModule,
)
from otter.model.vision.clip_model import (
    AllTokenFlaxCLIPModel,
    AllTokenFlaxClearCLIPModel,
)
from otter.model.vision.vit_encoders import normalize_images
import jax.numpy as jnp
from otter.model.transformer import (
    CrossAttentionPooling,
    TextImageCrossAttention,
)

from otter.model.vision.weight_loaders import (
    clip_large_weights_loader,
    clip_base16_weights_loader,
    clip_base32_weights_loader,
)

EPS = 1e-6


class ClearCLIPTokenizer(nn.Module):
    source: str = "openai/clip-vit-large-patch14"
    num_text_tokens: int = 1
    fusion_mlp_dim: int = 512
    num_fusion_layers: int = 1
    num_readouts: int = 4
    num_cameras: int = 2
    num_text_readouts: int = 1
    text_ratio: int = 1
    use_learnable_temperature: bool = False
    add_pe: bool = False
    get_pe: bool = False
    encode_text: bool = True

    def setup(self):
        self.clip_model = AllTokenFlaxClearCLIPModel.from_pretrained(self.source)

        assert (
            self.fusion_mlp_dim % (self.num_readouts * self.num_cameras) == 0
        ), f"mlp_dim must be divisible by num_readouts * num_cameras, got mlp_dim={self.mlp_dim}"
        for i in range(self.num_cameras):
            # create textimage cross attention and attention pooling for each camera
            setattr(
                self,
                f"text_image_pooling_{i}",
                TextImageCrossAttention(
                    use_learnable_temperature=self.use_learnable_temperature,
                    add_pe=self.add_pe,
                    get_pe=self.get_pe,
                ),
            )
            setattr(
                self,
                f"image_attn_pool_{i}",
                CrossAttentionPooling(
                    mlp_dim=self.fusion_mlp_dim,
                    out_dim=self.fusion_mlp_dim
                    // self.num_readouts
                    // self.num_cameras,  # this makes the combined output dim=512
                    use_query_tokens=True,
                    num_readouts=self.num_readouts,
                    num_layers=self.num_fusion_layers,
                    add_pe=False,
                    dropout_rate=0.0,
                ),
            )
        if self.encode_text:
            self.text_attn_pool = CrossAttentionPooling(
                mlp_dim=self.fusion_mlp_dim,
                out_dim=self.fusion_mlp_dim
                // self.num_text_readouts
                // self.num_cameras
                // self.text_ratio,
                use_query_tokens=True,
                num_readouts=self.num_text_readouts,
                num_layers=self.num_fusion_layers,
                add_pe=False,
                dropout_rate=0.0,
            )

    def encode_image(self, pixel_values, train):
        b, t, h, w, c = pixel_values.shape
        pixel_values = pixel_values.reshape((b * t, h, w, c))
        pixel_values = normalize_images(pixel_values, img_norm_type="clip")
        pixel_values = jnp.transpose(pixel_values, (0, 3, 1, 2))
        patch_tokens, cls_token = self.clip_model.get_image_features(pixel_values)
        return patch_tokens, cls_token

    def get_img_features(self, pixel_values, train):
        pixel_values = jnp.transpose(pixel_values, (0, 3, 1, 2))
        patch_tokens, cls_token = self.clip_model.get_image_features(pixel_values)
        return patch_tokens, cls_token

    def get_language_features(self, instruction, train):
        all_text_tokens, text_repr_tokens = self.clip_model.get_text_features(
            **instruction, train=train
        )
        return all_text_tokens, text_repr_tokens

    def __call__(self, observations, task, train: bool = True):
        # get language representations
        b, t = task["language_instruction"]["input_ids"].shape[:2]
        instruction = {}
        for k, v in task["language_instruction"].items():
            instruction[k] = v.copy().reshape((b * t, *v.shape[2:]))

        all_text_tokens, text_repr_tokens = self.clip_model.get_text_features(
            **instruction
        )

        all_text_tokens = all_text_tokens / jnp.linalg.norm(
            all_text_tokens, axis=-1, keepdims=True
        )

        # get observation representations
        all_image_cls_tokens = []
        all_image_patch_tokens = []
        available_image_keys = [k for k in observations.keys() if "image" in k]
        # sort the keys so that the order is consistent
        available_image_keys = sorted(available_image_keys)

        for k in available_image_keys:
            image = observations[k]
            image_patch_token, image_cls_token = self.encode_image(image, train=train)
            image_patch_token = image_patch_token / jnp.linalg.norm(
                image_patch_token, axis=-1, keepdims=True
            )
            all_image_cls_tokens.append(image_cls_token)
            all_image_patch_tokens.append(image_patch_token)

        # perform text image pooling for each image
        text_image_pooling = []
        for i in range(len(all_image_patch_tokens)):
            camera_i_textimg = getattr(self, f"text_image_pooling_{i}")(
                all_text_tokens, all_image_patch_tokens[i]
            )  # B, 16, 512
            camera_i_attnpool = getattr(self, f"image_attn_pool_{i}")(
                camera_i_textimg, train=train
            )  # B, 4, 64
            camera_i_attnpool = camera_i_attnpool.reshape(
                (b * t, -1)
            )  # collapse the last two dim
            text_image_pooling.append(camera_i_attnpool)

        # perform text pooling
        if self.encode_text:
            text_tokens = self.text_attn_pool(all_text_tokens, train=train).reshape(
                (b * t, -1)
            )
            text_image_pooling.append(text_tokens)
        text_image_token = jnp.concatenate(text_image_pooling, axis=-1).reshape(
            (b, t, -1)
        )
        return text_image_token


class MLP(nn.Module):
    output_dim: int
    hidden_size: int = 512
    n_layers: int = 2

    @nn.compact
    def __call__(self, input_tensor):
        assert self.n_layers > 0
        x = input_tensor

        for _ in range(self.n_layers - 1):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.relu(x)
        return nn.Dense(self.output_dim)(x)


class ProprioTokenizer(nn.Module):
    mlp_kwargs: dict = None
    encode_proprio: bool = True

    def setup(self):
        self.mlp = MLP(**self.mlp_kwargs)

    @nn.compact
    def __call__(
        self,
        observations,
        task=None,
        train: bool = True,
    ):
        b, t = observations["image_primary"].shape[:2]
        if self.encode_proprio and "proprio" in observations:
            state = observations["proprio"]
            state = jnp.reshape(state, (b * t, state.shape[-1]))
            state_tokens = self.mlp(state)
            state_tokens = state_tokens.reshape((b, t, -1))
        else:
            state_tokens = jnp.ones((b, t, self.mlp_kwargs["output_dim"]))
        return state_tokens


tokenizers = {
    "proprio-tokenizer": ProprioTokenizer,
    "clear-clip-tokenizer": ClearCLIPTokenizer,
}

# TODO this belongs somewhere else
weights_loaders = {
    "clip-large-loader": clip_large_weights_loader,
    "clip-base16-loader": clip_base16_weights_loader,
    "clip-base32-loader": clip_base32_weights_loader,
}
