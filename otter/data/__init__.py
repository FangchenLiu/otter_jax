from .dataset import icrt_restructure
from .oxe import OXE_NAMED_MIXES

DATASET_MAPPING = {
    "icrt": icrt_restructure,
}

OXE_NAMES = list(OXE_NAMED_MIXES.keys())
