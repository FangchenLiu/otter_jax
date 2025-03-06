# WARNING: importing tensorflow too late can silence important logging (╯°□°)╯︵ ┻━┻
import tensorflow as tf
import datetime
from functools import partial
import os
import os.path as osp
import numpy as np

from absl import app, flags, logging
from flax.traverse_util import flatten_dict
import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags
import optax
import tqdm
import wandb
from copy import deepcopy

import otter
from otter.model.otter_model import OtterModel
from otter.utils import jax_utils
from otter.utils.spec import ModuleSpec
from otter.utils.train_utils import (
    create_optimizer,
    format_name_with_config,
    process_text,
    Timer,
    TrainState,
    merge_params,
    check_config_diff,
)
from otter.utils.train_callbacks import SaveCallback
from otter.utils.typing import Data
from pathlib import Path
from otter.data.dataset import (
    make_single_dataset,
    make_interleaved_dataset,
)
from otter.data.oxe import make_oxe_dataset_kwargs_and_weights
from otter.data import DATASET_MAPPING, OXE_NAMES
from otter.data.utils.data_utils import NormalizationType
from otter.data.utils.text_processing import text_processor_dict
from otter.model.tokenizers import weights_loaders


FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

config_dir = os.path.join(os.path.dirname(__file__), "configs")
config_flags.DEFINE_config_file(
    "config",
    os.path.join(config_dir, "train_config.py"),
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def create_dataset(train, dataset_name=None):
    dataset_kwargs_list = []
    sample_weights = []
    dataset_kwargs = deepcopy(FLAGS.config.dataset.dataset_kwargs)
    for data_kwargs in dataset_kwargs:
        if dataset_name is not None and data_kwargs["name"] not in dataset_name:
            continue

        if data_kwargs["name"] in OXE_NAMES:
            oxe_dataset_kwargs_list, weights = make_oxe_dataset_kwargs_and_weights(
                data_kwargs["name"],
                data_kwargs["data_dir"],
                force_recompute_dataset_statistics=False,
            )
            for k, v in data_kwargs.items():
                for oxe_dataset_kwargs in oxe_dataset_kwargs_list:
                    if k not in oxe_dataset_kwargs:
                        oxe_dataset_kwargs[k] = v
                    oxe_dataset_kwargs["action_horizon"] = (
                        FLAGS.config.model.policy_kwargs.action_pred_horizon
                    )

            dataset_kwargs_list.extend(oxe_dataset_kwargs_list)
            sample_weights.extend(weights)

        else:
            data_kwargs["action_horizon"] = (
                FLAGS.config.model.policy_kwargs.action_pred_horizon
            )
            if data_kwargs["action_proprio_normalization_type"] == "normal":
                data_kwargs["action_proprio_normalization_type"] = (
                    NormalizationType.NORMAL
                )

            for k, v in DATASET_MAPPING.items():
                if k in data_kwargs["name"]:
                    data_kwargs["restructure"] = v
                    break

            dataset_kwargs_list.append(data_kwargs)

    if "sample_weights" in FLAGS.config.dataset:
        sample_weights.extend(FLAGS.config.dataset["sample_weights"])

    dataset = make_interleaved_dataset(
        dataset_kwargs_list=dataset_kwargs_list,
        sample_weights=sample_weights,
        train=train,
        traj_transform_kwargs=FLAGS.config.dataset.traj_transform_kwargs,
        frame_transform_kwargs=FLAGS.config.dataset.frame_transform_kwargs,
        batch_size=FLAGS.config.batch_size,
        **FLAGS.config.dataset.others,
    )

    FLAGS.config.model.policy_kwargs["action_dim"] = len(
        dataset_kwargs[0]["action_normalization_mask"]
    )
    FLAGS.config.model.policy_kwargs["time_sequence_length"] = (
        FLAGS.config.dataset.traj_transform_kwargs["subsample_length"]
    )
    return dataset


def main(_):
    jax_utils.initialize_compilation_cache()

    assert FLAGS.config.batch_size % jax.device_count() == 0
    assert FLAGS.config.batch_size % jax.process_count() == 0

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # replicated sharding -- does not shard arrays
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    # data-parallel sharding -- shards arrays along the first axis
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))

    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(
            batch, mesh, PartitionSpec("batch")
        )

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # make sure each process loads different data
    tf.random.set_seed(FLAGS.config.seed + jax.process_index())

    # set up wandb and logging
    name = format_name_with_config(
        FLAGS.name,
        FLAGS.config.to_dict(),
    )
    wandb_id = "{name}_{time}".format(
        name=name,
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb_id = jax_utils.host_broadcast_str(wandb_id)
    if jax.process_index() == 0:
        wandb.init(
            config=FLAGS.config.to_dict(),
            id=wandb_id,
            name=name,
            mode="disabled" if FLAGS.debug else None,
            **FLAGS.config.wandb,
        )

    if FLAGS.config.save_dir is not None:
        save_dir = tf.io.gfile.join(
            FLAGS.config.save_dir,
            FLAGS.config.wandb.project,
            FLAGS.config.wandb.group or "",
            wandb_id,
        )
        logging.info("Saving to %s", save_dir)
        if jax.process_index() == 0:
            wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
    else:
        save_dir = None
        logging.info("save_dir not passed in, not saving checkpoints")

    pretrained_model = None
    if FLAGS.config.get("wandb_resume_id", None) is not None:
        # resume previous run
        wandb_run = wandb.Api().run(FLAGS.config.wandb_resume_id)
        pretrained_dir = wandb_run.config["save_dir"]
        logging.info("Resuming run %s", FLAGS.config.wandb_resume_id)
        pretrained_model = OtterModel.load_pretrained(pretrained_dir)

    save_callback = SaveCallback(save_dir)

    if jax.process_index() == 0:
        codebase_directory = osp.abspath(osp.join(osp.dirname(otter.__file__), ".."))
        wandb.run.log_code(codebase_directory)

    # set up text tokenization (this needs to happen after batching but before sharding)
    if FLAGS.config.text_processor is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.create(
            text_processor_dict[FLAGS.config.text_processor]
        )
        text_processor = ModuleSpec.instantiate(text_processor)(
            **FLAGS.config.text_processor_kwargs
        )

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    FLAGS.config.batch_size //= jax.process_count()
    train_data = create_dataset(True)

    if FLAGS.config.dataset.eval_data is not None:
        if isinstance(FLAGS.config.dataset.eval_data, str):
            eval_data_name = [FLAGS.config.dataset.eval_data]
        else:
            eval_data_name = FLAGS.config.dataset.eval_data

        eval_data = create_dataset(False, eval_data_name)
    else:
        eval_data = None

    train_data_iter = map(
        shard,
        map(
            process_batch,
            train_data.iterator(prefetch=FLAGS.config.dataset.prefetch_num_batches),
        ),
    )

    if eval_data is not None:
        eval_data_iter = map(
            shard,
            map(
                process_batch,
                eval_data.iterator(),
            ),
        )

    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['action'].shape[0]}")
    logging.info(f"Number of devices: {jax.device_count()}")
    logging.info(
        f"Batch size per device: {example_batch['action'].shape[0] // jax.device_count()}"
    )

    # set up model and initialize weights
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)
    model = OtterModel.from_config(
        FLAGS.config.to_dict(),
        example_batch,
        text_processor,
        verbose=True,
        rng=init_rng,
        dataset_statistics=train_data.dataset_statistics,
    )
    if pretrained_model is not None:
        print("Loading and checking the difference with pretrained model")
        check_config_diff(FLAGS.config.to_dict(), pretrained_model.config)
        merged_params = merge_params(model.params, pretrained_model.params)
        model = model.replace(params=merged_params)
        del pretrained_model

    # create optimizer
    tx, lr_callable, param_norm_callable = create_optimizer(
        model.params,
        **FLAGS.config.optimizer.to_dict(),
    )

    # Load pretrained weights (e.g. text encoder) if necessary
    for loader_name in FLAGS.config.pretrained_loaders:
        if not callable(loader_name):  # Means that it is a ModuleSpec
            loader = weights_loaders[loader_name]
            loader = ModuleSpec.create(loader)
            loader = ModuleSpec.instantiate(loader)
        model = model.replace(params=loader(model.params))

    # create train state
    train_state = TrainState.create(rng, model, tx)
    start_step = FLAGS.config.start_step or 0
    train_state = train_state.replace(step=start_step)
    # refreshes the train state so it doesn't crash w/ certain pre-trained loaders
    train_state = jax.device_get(train_state)

    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        info = bound_module(
            batch["observation"],
            batch["task"],
            batch["action"],
            train=train,
        )
        return info["loss"], info

    @partial(
        jax.jit,
        # state is replicated, batch is data-parallel
        in_shardings=(replicated_sharding, dp_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
        # allows jax to modify `state` in-place, saving a lot of memory
        donate_argnums=0,
    )
    def train_step(state: TrainState, batch: Data):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        grad_norm = optax.global_norm(grads)
        updates, _ = state.tx.update(grads, state.opt_state, state.model.params)
        update_norm = optax.global_norm(updates)
        info.update(
            {
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "param_norm": param_norm_callable(state.model.params),
                "learning_rate": lr_callable(state.step),
            }
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    @jax.jit
    def eval_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        loss, info = loss_fn(state.model.params, batch, dropout_rng, train=False)
        return info

    def wandb_log(info, step):
        if jax.process_index() == 0:
            wandb.log(flatten_dict(info, sep="/"), step=step)

    num_eval_batches = 10
    timer = Timer()
    for i in tqdm.tqdm(
        range(start_step, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps),
        initial=start_step,
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        if (i + 1) % FLAGS.config.save_interval == 0:
            save_callback(train_state, i + 1)

        # if (i + 1) % FLAGS.config.eval_interval == 0:
        if (i) % FLAGS.config.eval_interval == 0:

            logging.info("Evaluating...")
            with timer("eval"):
                metrics = []
                for _ in tqdm.tqdm(range(num_eval_batches)):

                    if eval_data is not None:
                        eval_batch = next(eval_data_iter)
                        eval_metrics = eval_step(train_state, eval_batch)
                        metrics.append({"loss": eval_metrics["loss"]})
                if eval_data is not None:
                    metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
                    wandb_log({"val/validation": metrics}, step=i)

        timer.tock("total")
        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()},
                step=i + 1,
            )


if __name__ == "__main__":
    app.run(main)
