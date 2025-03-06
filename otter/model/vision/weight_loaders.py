import flax.linen as nn
import jax
from flax.core.frozen_dict import freeze
from otter.model.vision.clip_model import (
    AllTokenFlaxCLIPModel,
    AllTokenFlaxClearCLIPModel,
)


# after model intialization, call this to load CLIP weights into params
def clip_weights_loader(params, source):
    clip = AllTokenFlaxClearCLIPModel.from_pretrained(source)
    clip_def, clip_variables = clip.module, clip.params
    clip_params = clip_variables

    def find_and_replace(params, key, replacement):
        for k in params.keys():
            if k == key:
                params[k] = replacement
                print(f"Replaced {key} in params")
                return
            if isinstance(params[k], type(params)):
                find_and_replace(params[k], key, replacement)

    # params = params.unfreeze()
    find_and_replace(params, "clip_model", clip_params)
    return params


def clip_base16_weights_loader(params):
    return clip_weights_loader(params, "openai/clip-vit-base-patch16")


def clip_base32_weights_loader(params):
    return clip_weights_loader(params, "openai/clip-vit-base-patch32")


def clip_large_weights_loader(params):
    return clip_weights_loader(params, "openai/clip-vit-large-patch14")
