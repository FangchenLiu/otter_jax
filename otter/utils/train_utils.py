from collections import defaultdict
from contextlib import contextmanager
from fnmatch import fnmatch
import logging
import time
from typing import Callable, List, Optional
import jax
import jax.numpy as jnp
import optax
from flax import serialization
import flax
from flax.training import train_state
from jax.experimental.compilation_cache import compilation_cache
from otter.data.utils.text_processing import TextProcessor
from otter.utils.typing import Data, Params, PRNGKey, Config
from otter.data.utils.utils import action_delta2abs, project2cam
from otter.data.utils.data_utils import denormalize, NormalizationType
from flax.core.frozen_dict import FrozenDict, freeze
from flax import struct
from otter.model.otter_model import OtterModel
from ml_collections import ConfigDict

import numpy as np

TASKS = {
    "Pick and place tiger into the black bowl": [
        "Pick and place tiger into the black bowl",
        "Pick up the tiger and place it in the black bowl",
        "Put the tiger in the black bowl",
    ],
    "Stack an orange cup on a yellow cup": [
        "Stack an orange cup on a yellow cup",
        "Pick up and place an orange cup in a yellow cup",
        "Pick up the orange cup and stack it in the yellow cup",
        "Pick up the orange cup and place it in the yellow cup",
        "Put the orange cup on the yellow cup",
    ],
    "Open a drawer": ["Open a drawer"],
    "Poke a block": ["Poke a block"],
    "Push a blue cube": ["Push a blue cube"],
    "Put a potato in a black bowl": [
        "Put a potato in a black bowl",
        "Pick up a potato and place it in the black bowl",
        "Pick up and place the potato in the black bowl",
    ],
    "Pick up a potato": ["Pick up a potato", "Grasp a potato"],
    "Stack an orange cup on a yellow cup": [
        "Stack an orange cup on a yellow cup",
        "Pick up and place an orange cup inon a yellow cup",
        "Pick up the orange cup and stack it in the yellow cup",
        "Pick up the orange cup and place it in the yellow cup",
        "Put the orange cup on the yellow cup",
    ],
    "Pick and place a deer in a grey bowl": [
        "Pick and place a deer in a grey bowl",
        "Pick up the deer and place it in the grey bowl",
        "Put the deer in the grey bowl",
    ],
    "Pick up a green triangle": ["Pick up a green triangle"],
    "Stack a red block on a green block": [
        "Stack a red block on a green block",
        "Pick up the red block and stack it on the green block",
        "Pick up the red block and place it on the green block",
        "Put the red block on the green block",
        "Pick up and place the red block on the green block",
    ],
    "Put a tiger in a black bowl": [
        "Put a tiger in a black bowl",
        "Pick up the tiger and place it in the black bowl",
        "Put up and place the tiger in the black bowl",
    ],
    "Poke a tiger": ["Poke a tiger"],
    "Pick and place a red cube into a black bowl": [
        "Pick and place a red cube into a black bowl",
        "Pick up the red cube and place it in the black bowl",
        "Put the red cube in the black bowl",
    ],
    "Pick a blue cube and stack it on a wood block": [
        "Pick a blue cube and stack it on a wood block",
        "Pick up the blue cube and stack it on the wood block",
        "Put the blue cube on the wood block",
        "Pick up and place the blue cube on the wood block",
    ],
    "Pick and place a blue cube into a grey bowl": [
        "Pick and place a blue cube into a grey bowl",
        "Pick up the blue cube and place it in the grey bowl",
        "Put the blue cube in the grey bowl",
    ],
    "Put a red ball in a black bowl": [
        "Put a red ball in a black bowl",
        "Pick up the red ball and place it in the black bowl",
        "Put the red ball in the black bowl",
    ],
    "Pick and place a green triangle into a pink bowl": [
        "Pick and place a green triangle into a pink bowl",
        "Pick up the green triangle and place it in the pink bowl",
        "Put the green triangle in the pink bowl",
    ],
    "Pick and place a red ball into a pink bowl": [
        "Pick and place a red ball into a pink bowl",
        "Pick up the red ball and place it in the pink bowl",
        "Put the red ball in the pink bowl",
    ],
    "Poke a green triangle": ["Poke a green triangle"],
    "Poke a grey bowl": ["Poke a grey bowl"],
    "Put a blue cube in a pink bowl": [
        "Put a blue cube in a pink bowl",
        "Pick up the blue cube and place it in the pink bowl",
        "Put the blue cube in the pink bowl",
    ],
    "Stack a red block on a cyan block": [
        "Stack a red block on a cyan block",
        "Pick up the red block and stack it on the cyan block",
        "Pick up the red block and place it on the cyan block",
        "Put the red block on the cyan block",
        "Pick up and place the red block on the cyan block",
    ],
    "Stack a yellow block on a black block": [
        "Stack a yellow block on a black block",
        "Pick up the yellow block and stack it on the black block",
        "Pick up the yellow block and place it on the black block",
        "Put the yellow block on the black block",
        "Pick up and place the yellow block on the black block",
    ],
    "Stack a black block on a red block": [
        "Stack a black block on a red block",
        "Pick up the black block and stack it on the red block",
        "Pick up the black block and place it on the red block",
        "Put the black block on the red block",
        "Pick up and place the black block on the red block",
    ],
    "Pick and place the deer in a grey bowl": [
        "Pick and place the deer in a grey bowl",
        "Pick up the deer and place it in the grey bowl",
        "Put the deer in the grey bowl",
    ],
}


def process_text(batch: Data, text_processor: Optional[TextProcessor]) -> Data:
    """Encodes the language instruction inside the tasks for a batch.

    If the text processor is None, removes language entirely from the tasks.
    Expects batch to be a nested dictionary, where
        batch["task"]["language_instruction"] is a sequence of byte strings
    """
    if text_processor is None:
        batch["task"].pop("language_instruction")
    else:

        instruction = batch["task"]["language_instruction"]
        all_instruction = [instruction]

        instruction2 = batch["task"].get("language_instruction_2")
        instruction3 = batch["task"].get("language_instruction_3")

        if instruction2 is not None:
            all_instruction.append(instruction2)
        if instruction3 is not None:
            all_instruction.append(instruction3)

        instruction_idx = np.random.choice(len(all_instruction))
        instruction = all_instruction[instruction_idx]

        if instruction2:
            del batch["task"]["language_instruction_2"]
        if instruction3:
            del batch["task"]["language_instruction_3"]

        decodes = []
        for seq in instruction:
            decode_s = seq[0].decode("utf-8")
            if decode_s in TASKS:
                decodes.extend([np.random.choice(TASKS[decode_s])] * len(seq))
            else:
                for s in seq:
                    decodes.append(s.decode("utf-8"))

        batch["task"]["language_instruction"] = text_processor.encode(decodes)
        if isinstance(batch["task"]["language_instruction"], np.ndarray):
            batch["task"]["language_instruction"] = batch["task"][
                "language_instruction"
            ].reshape(instruction.shape[0], instruction.shape[1], -1)
        else:
            batch["task"]["language_instruction"] = {
                k: v.reshape(instruction.shape[0], instruction.shape[1], -1)
                for k, v in batch["task"]["language_instruction"].items()
            }

        # print(batch["task"]["language_instruction"]['position_ids'].shape)
        # batch["task"]["language_instruction"] = text_processor.encode(
        #     [s.decode("utf-8") for seq in batch["task"]["language_instruction"]]
        # )
    return batch


@struct.dataclass
class TrainState:
    rng: PRNGKey
    model: OtterModel
    step: int
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        model: OtterModel,
        tx: optax.GradientTransformation,
    ):
        opt_state = tx.init(model.params)
        return cls(
            rng=rng,
            model=model,
            step=0,
            opt_state=opt_state,
            tx=tx,
        )

    def apply_gradients(self, *, grads, rng):
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.model.params
        )
        new_params = optax.apply_updates(self.model.params, updates)

        return self.replace(
            step=self.step + 1,
            model=self.model.replace(params=new_params),
            opt_state=new_opt_state,
            rng=rng,
        )


def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax Sharding object with shape (num_devices,).
    """
    return jax.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


def create_train_state(
    rng,
    model_def,
    optimizer_kwargs=(),
    init_args=(),
    init_kwargs=dict(),
    pretrained_loaders=tuple(),
):
    """Utility to create a TrainState."""
    init_rng, state_rng = jax.random.split(rng)

    # Initializing the model in a jit avoids running the model on CPU
    @jax.jit
    def init(rng):
        return model_def.init(rng, *init_args, **init_kwargs)

    ev, params = flax.core.pop(init(init_rng), "params")
    params = freeze(params)

    # create optimizer
    tx, lr_callable, param_norm_callable = create_optimizer(params, **optimizer_kwargs)

    assert (
        len(ev) == 0
    ), "Are you forgetting to store some variables in the state? {}".format(ev.keys())

    for loader in pretrained_loaders:
        params = loader(params)

    return TrainState.create(
        apply_fn=model_def.apply,
        params=params,
        tx=tx,
        rng=state_rng,
    )


def format_name_with_config(name, config):
    """Formats a name string with a config dict.

    Formatting keys may be specified as {key} or {full_path_to_key_with_underscores}.

    Example:
        name = "model_{model_type}_{model_size}"
        config = {"model_type": "transformer", "model_size": "small"}
        format_name_with_config(name, config) -> "model_transformer_small"
    """
    config_flat = flax.traverse_util.flatten_dict(config, sep="_")
    config_final = {k.split("_")[-1]: v for k, v in config_flat.items()}
    format_dict = {**config_final, **config_flat}
    return name.format(**format_dict)


class Timer:
    """
    Timer utility. Usage:

        timer = Timer()
        with timer("foo"):
            do_something()

        timer.tick("bar")
        do_something_else()
        timer.tock("bar")

        timer.get_average_times() -> {"foo": 0.1, "bar": 0.2}
    """

    def __init__(self):
        self.reset()

    @contextmanager
    def __call__(self, key):
        self.tick(key)
        try:
            yield None
        finally:
            self.tock(key)

    def reset(self):
        self.counts = defaultdict(int)
        self.times = defaultdict(float)
        self.start_times = {}

    def tick(self, key):
        if key in self.start_times:
            raise ValueError(f"Timer is already ticking for key: {key}")
        self.start_times[key] = time.time()

    def tock(self, key):
        if key not in self.start_times:
            raise ValueError(f"Timer is not ticking for key: {key}")
        self.counts[key] += 1
        self.times[key] += time.time() - self.start_times[key]
        del self.start_times[key]

    def get_average_times(self, reset=True):
        ret = {key: self.times[key] / self.counts[key] for key in self.counts}
        if reset:
            self.reset()
        return ret


def initialize_compilation_cache(cache_dir="/tmp/jax_cache"):
    compilation_cache.initialize_cache(cache_dir)

    for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]:
        logger.addFilter(
            lambda record: "Not writing persistent cache entry for"
            not in record.getMessage()
        )


def create_lr_schedule(name: str, **kwargs):
    """Creates a learning rate callable.

    Currently supported schedules:
        cosine: cosine decay with warmup.
            kwargs: init_value, peak_value, warmup_steps, decay_steps
        rsqrt: inverse square root decay with warmup, from the "Scaling Vision Transformers" paper.
            kwargs: init_value, peak_value, warmup_steps, timescale (optional, default 10000)
        constant: constant learning rate with warmup.
            kwargs: init_value, peak_value, warmup_steps

    Args:
        name: name of the schedule
        **kwargs: additional kwargs, which vary by schedule
    """
    if name == "cosine":
        return optax.warmup_cosine_decay_schedule(**kwargs)
    elif name == "rsqrt":
        timescale = kwargs.get("timescale", 10000)
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=kwargs["init_value"],
                    end_value=kwargs["peak_value"],
                    transition_steps=kwargs["warmup_steps"],
                ),
                lambda step: kwargs["peak_value"]
                / jnp.sqrt((step + timescale) / timescale),
            ],
            [kwargs["warmup_steps"]],
        )
    elif name == "constant":
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=kwargs["init_value"],
                    end_value=kwargs["peak_value"],
                    transition_steps=kwargs["warmup_steps"],
                ),
                lambda step: kwargs["peak_value"],
            ],
            [kwargs["warmup_steps"]],
        )
    else:
        raise ValueError(f"Unsupported lr schedule: {name}")


def merge_params(target_params: Params, pretrained_params: Params) -> Params:
    """Copies pre-trained params into target_params for every param that has corresponding key + shape."""
    flat_target_params = flax.traverse_util.flatten_dict(target_params)
    flat_pretrained_params = flax.traverse_util.flatten_dict(pretrained_params)
    keys_to_update = [
        k
        for k in flat_target_params
        if k in flat_pretrained_params
        and flat_target_params[k].shape == flat_pretrained_params[k].shape
    ]
    missing_keys = [k for k in flat_target_params if k not in flat_pretrained_params]
    shape_mismatch_keys = [
        k
        for k in flat_target_params
        if k in flat_pretrained_params
        and flat_target_params[k].shape != flat_pretrained_params[k].shape
    ]

    for key in keys_to_update:
        logging.debug(f"Param copied from pre-trained: {'.'.join(key)}")
    if missing_keys or shape_mismatch_keys:
        logging.info("########## Parameters skipped during model loading: ##########")
        for key in missing_keys:
            logging.info(
                f"Param missing in pre-trained model, skipping: {'.'.join(key)}"
            )
        for key in shape_mismatch_keys:
            logging.info(
                f"Param with differing shape in pre-trained model, skipping: {'.'.join(key)}"
            )

    flat_target_params = flax.core.copy(
        flat_target_params, {k: flat_pretrained_params[k] for k in keys_to_update}
    )
    target_params = flax.traverse_util.unflatten_dict(flat_target_params)
    return target_params


def check_config_diff(new_conf: Config, old_conf: Config, silent: bool = False):
    """Checks for differences between new config and old config dicts."""
    new_conf_flat = flax.traverse_util.flatten_dict(
        new_conf.to_dict() if isinstance(new_conf, ConfigDict) else new_conf
    )
    old_conf_flat = flax.traverse_util.flatten_dict(
        old_conf.to_dict() if isinstance(old_conf, ConfigDict) else old_conf
    )

    # check for missing / new keys
    if set(new_conf_flat.keys()) != set(old_conf_flat.keys()) and not silent:
        logging.info(
            "New config contains extra items: %s",
            set(new_conf_flat.keys()) - set(old_conf_flat.keys()),
        )
        logging.info(
            "New config doesn't contain items: %s",
            set(old_conf_flat.keys()) - set(new_conf_flat.keys()),
        )

    # print differing key values
    mismatched_keys = {
        k: (new_conf_flat[k], old_conf_flat[k])
        for k in new_conf_flat
        if k in old_conf_flat and new_conf_flat[k] != old_conf_flat[k]
    }
    if mismatched_keys and not silent:
        logging.info(
            "New config contains keys with new values: %s",
            flax.core.pretty_repr(mismatched_keys),
        )
    return mismatched_keys or (set(new_conf_flat.keys()) != set(old_conf_flat.keys()))


def freeze_weights(
    tx: optax.GradientTransformation,
    params_or_params_shape: Params,
    frozen_keys: List[str],
    return_partitions: bool = False,
):
    """
    Freezes all weights in params_or_params_shape whose keys fnmatch the ones in frozen_keys.
    Example usage:
        tx = freeze_weights(tx, model.params, ["octo_transformer.*"])
    """
    logging.info(f"Freezing parameters that include the following keys: {frozen_keys}.")
    partition_optimizers = {
        "trainable": tx,
        "frozen": optax.set_to_zero(),
    }
    # freeze anything that matches fnmatch patterns in `frozen_keys`
    # path is a string of .-separated module names, e.g. ('octo_transformer.BlockTransformer_0...')
    param_partitions = flax.traverse_util.path_aware_map(
        lambda path, v: (
            "frozen"
            if any([fnmatch(".".join(path), key) for key in frozen_keys])
            else "trainable"
        ),
        params_or_params_shape,
    )

    # # print trainable parameters
    # for k, v in flax.traverse_util.flatten_dict(param_partitions).items():
    #     if v == 'trainable':
    #         print(k)

    tx = optax.multi_transform(partition_optimizers, param_partitions)

    logging.debug("Frozen params:")
    flax.traverse_util.path_aware_map(
        lambda path, opt_status: (
            logging.debug(".".join(path)) if opt_status == "frozen" else None
        ),
        param_partitions,
    )
    total_params = sum(
        jax.tree_util.tree_leaves(
            jax.tree_map(lambda x: x.size, params_or_params_shape)
        )
    )
    trainable_params = sum(
        jax.tree_util.tree_leaves(
            jax.tree_map(
                lambda x, y: x.size if y == "trainable" else 0,
                params_or_params_shape,
                param_partitions,
            )
        )
    )
    logging.info(f"Num trainable params: {trainable_params:,}.")
    logging.info(f"Num frozen params: {total_params - trainable_params:,}.")
    logging.info("To see a detailed list of frozen params, set logging level to DEBUG.")
    return (tx, param_partitions) if return_partitions else tx


def unfreeze_weights(
    tx: optax.GradientTransformation,
    params_or_params_shape: Params,
    trainable_keys: List[str],
    return_partitions: bool = False,
):
    """
    Freezes all weights in params_or_params_shape whose keys fnmatch the ones in trainable_keys.
    Example usage:
        tx = freeze_weights(tx, model.params, ["octo_transformer.*"])
    """
    logging.info(
        f"UnFreezing parameters that include the following keys: {trainable_keys}."
    )
    partition_optimizers = {
        "trainable": tx,
        "frozen": optax.set_to_zero(),
    }
    # freeze anything that matches fnmatch patterns in `trainable_keys`
    # path is a string of .-separated module names, e.g. ('octo_transformer.BlockTransformer_0...')
    param_partitions = flax.traverse_util.path_aware_map(
        lambda path, v: (
            "trainable"
            if any([fnmatch(".".join(path), key) for key in trainable_keys])
            else "frozen"
        ),
        params_or_params_shape,
    )

    # print('param_partitions:', param_partitions)
    # # print trainable parameters
    # for k, v in flax.traverse_util.flatten_dict(param_partitions).items():
    #     if v == 'trainable':
    #         print(k)

    tx = optax.multi_transform(partition_optimizers, param_partitions)

    logging.debug("Frozen params:")
    flax.traverse_util.path_aware_map(
        lambda path, opt_status: (
            logging.debug(".".join(path)) if opt_status == "frozen" else None
        ),
        param_partitions,
    )
    total_params = sum(
        jax.tree_util.tree_leaves(
            jax.tree_map(lambda x: x.size, params_or_params_shape)
        )
    )
    trainable_params = sum(
        jax.tree_util.tree_leaves(
            jax.tree_map(
                lambda x, y: x.size if y == "trainable" else 0,
                params_or_params_shape,
                param_partitions,
            )
        )
    )
    logging.info(f"Num trainable params: {trainable_params:,}.")
    logging.info(f"Num frozen params: {total_params - trainable_params:,}.")
    logging.info("To see a detailed list of frozen params, set logging level to DEBUG.")
    return (tx, param_partitions) if return_partitions else tx


def create_optimizer(
    params_or_params_shape: Params, **kwargs: dict
) -> optax.GradientTransformation:
    """Creates optimizer for Octo.

    kwargs are the kwargs for optax.adamw; if the "learning_rate" key is a dict, it is interpreted
    as the kwargs for create_lr_schedule (see above), otherwise it is interpreted as a constant
    learning rate.

    If clip_gradient is specified, then gradient clipping is applied. If frozen_keys is specified,
    then those parameters are frozen (i.e. not updated) during training.

    Returns:
        tx: an Optax optimizer
        lr_callable: Function that takes the current step and returns the learning rate
    """
    if isinstance(kwargs["learning_rate"], dict):
        lr_callable = create_lr_schedule(**kwargs["learning_rate"])
    else:
        lr_callable = lambda _: kwargs["learning_rate"]
    kwargs["learning_rate"] = lr_callable

    # Following ViT, timm, MAE: this mask skips weight decay on biases and LayerNorm parameters
    wd_mask = jax.tree_util.tree_map_with_path(
        lambda path, x: "kernel" in jax.tree_util.keystr(path), params_or_params_shape
    )

    clip_gradient = kwargs.pop("clip_gradient", None)
    frozen_keys = kwargs.pop("frozen_keys", None)
    trainable_keys = kwargs.pop("trainable_keys", None)
    assert not (
        frozen_keys and trainable_keys
    ), "Cannot specify both frozen_keys and trainable_keys"
    grad_accumulation_steps = kwargs.pop("grad_accumulation_steps", None)

    tx = optax.adamw(mu_dtype=jnp.bfloat16, **kwargs, mask=wd_mask)
    if grad_accumulation_steps:
        tx = optax.MultiSteps(tx, grad_accumulation_steps)
    if clip_gradient is not None:
        tx = optax.chain(
            optax.clip_by_global_norm(clip_gradient),
            tx,
        )

    if frozen_keys:
        tx, param_partitions = freeze_weights(
            tx, params_or_params_shape, frozen_keys, return_partitions=True
        )
        zero_frozen_params = lambda params: jax.tree_map(
            lambda x, y: x if y == "trainable" else jnp.zeros(()),
            params,
            param_partitions,
        )
        param_norm_callable = lambda params: optax.global_norm(
            zero_frozen_params(params)
        )
    else:
        if trainable_keys:
            tx, param_partitions = unfreeze_weights(
                tx, params_or_params_shape, trainable_keys, return_partitions=True
            )
            zero_frozen_params = lambda params: jax.tree_map(
                lambda x, y: x if y == "trainable" else jnp.zeros(()),
                params,
                param_partitions,
            )
            param_norm_callable = lambda params: optax.global_norm(
                zero_frozen_params(params)
            )
        else:
            param_norm_callable = optax.global_norm

    return tx, lr_callable, param_norm_callable
