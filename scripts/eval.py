# WARNING: importing tensorflow too late can silence important logging (╯°□°)╯︵ ┻━┻
import tensorflow as tf

# isort: split

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
    plot_val,
)
from otter.utils.train_callbacks import SaveCallback
from otter.utils.typing import Data
from pathlib import Path
from otter.data.dataset import make_interleaved_dataset
from otter.data import DATASET_MAPPING
from otter.data.utils.data_utils import NormalizationType
from otter.data.utils.text_processing import text_processor_dict
from otter.model.tokenizers import weights_loaders
import matplotlib.pyplot as plt
from ml_collections import ConfigDict

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", None, "Path to finetuned Octo checkpoint directory."
)


def create_dataset(train, pretrained_config, dataset_name=None):
    dataset_kwargs_list = []
    dataset_kwargs = deepcopy(pretrained_config.dataset.dataset_kwargs)
    for data_kwargs in dataset_kwargs:
        if dataset_name is not None and data_kwargs["name"] not in dataset_name:
            continue
        data_kwargs["action_horizon"] = pretrained_config["model"]["policy_kwargs"][
            "action_pred_horizon"
        ]
        if data_kwargs["action_proprio_normalization_type"] == "normal":
            data_kwargs["action_proprio_normalization_type"] = NormalizationType.NORMAL

        for k, v in DATASET_MAPPING.items():
            if k in data_kwargs["name"]:
                data_kwargs["restructure"] = v
                break
        dataset_kwargs_list.append(data_kwargs)

    dataset = make_interleaved_dataset(
        dataset_kwargs_list=dataset_kwargs_list,
        train=train,
        traj_transform_kwargs=pretrained_config.dataset.traj_transform_kwargs,
        frame_transform_kwargs=pretrained_config.dataset.frame_transform_kwargs,
        batch_size=4,
        **pretrained_config.dataset.others,
    )

    return dataset


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


def undo_vision_transform(obs, mean: tuple, std: tuple):
    """
    Undo the vision transform applied to the observations.
    torch tensor has shape T, num_cam, 3, H, W
    return np.ndarray with shape T, num_cam * H, W, 3 at np.uint8
    """
    # undo normalization
    obs = obs.permute(0, 1, 3, 4, 2)
    obs = obs * std + mean
    obs = obs.numpy()
    obs = np.clip(obs * 255, 0, 255).astype(np.uint8)
    obs = np.concatenate([obs[:, i] for i in range(obs.shape[1])], axis=1)
    return obs


def main(_):
    # load finetuned model
    logging.info("Loading finetuned model...")
    tf.config.set_visible_devices([], "GPU")
    pretrained_model = OtterModel.load_pretrained(FLAGS.finetuned_path)
    mesh = Mesh(jax.devices(), axis_names="batch")

    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(
            batch, mesh, PartitionSpec("batch")
        )

    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    base_data_config = dict(
        dataset_kwargs=[
            dict(
                name="icrt_pickplace",
                data_dir="/home/fangchen/icrt_data/icrt_pickplace_clean/icrl_data_tfds/1.0.0",
                shuffle=True,
                action_normalization_mask=[True] * 9 + [False],
                skip_norm=True,
                action_proprio_normalization_type="normal",
                proprio_noise=0.02,
                # dataset_statistics="/dev/shm/icrl_octo_data_tfds/icrl_data_tfds/1.0.0/dataset_statistics_5747396208d1ffb0f90306ce9d8bfdc02ecf499b9336ee82cfc75657f6004a91.json"
            ),
        ],
        traj_transform_kwargs=dict(
            goal_relabeling_strategy=None,
            subsample_length=12,  # !!!!!!!!!!!!!!!!!!!!!!
            task_augment_strategy=None,
        ),
        frame_transform_kwargs=dict(
            resize_size=(224, 224),
            image_dropout_prob=0.0,
            image_augment_kwargs=dict(
                primary=dict(
                    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.1],
                    augment_order=[
                        # "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )
            ),
            num_parallel_calls=400,
        ),
        others=dict(
            shuffle_buffer_size=1000,  # !!!!!!!!!!!!!!!!!!!!!!
            traj_transform_threads=48,  # shared between all datasets
            traj_read_threads=48,  # shared between all datasets
        ),
        prefetch_num_batches=20,
        log_robot_data=[
            "icrt_pickplace",
            "icrt_stack",
            "icrt_0926",
        ],  # !!!!!!!!!!!!!!!!!!!!!!
        log_vis_data=None,  # !!!!!!!!!!!!!!!!!!!!!!
    )

    config = ConfigDict(pretrained_model.config)
    config.dataset = ConfigDict(base_data_config)
    eval_data = create_dataset(False, config, "icrt_pickplace")
    eval_data_iter = map(
        shard,
        map(
            process_batch,
            eval_data.iterator(),
        ),
    )

    eval_batch = next(eval_data_iter)

    # add timer
    import time

    tik = time.time()
    _ = pretrained_model.sample_actions(
        eval_batch["observation"],
        eval_batch["task"],
    )
    tok = time.time()
    print(f"Time taken for otter model action prediction: {tok - tik}")

    # change finetune encoder
    # print(config["model"]['task_tokenizer_kwargs']['clean-clip-multi-level-tokenizer']['finetune_encoder'])
    config["model"]["task_tokenizer_kwargs"]["clean-clip-multi-level-tokenizer"][
        "finetune_encoder"
    ] = False
    print("change to hugging face encoder")
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    new_model = OtterModel.from_config(
        config,
        eval_batch,
        text_processor=text_processor,
        verbose=True,
        rng=init_rng,
        dataset_statistics=eval_data.dataset_statistics,
    )
    print("hf new model created")

    tik = time.time()
    _ = new_model.sample_actions(
        eval_batch["observation"],
        eval_batch["task"],
    )
    tok = time.time()
    print(f"Time taken for hf new model action prediction: {tok - tik}")


if __name__ == "__main__":
    app.run(main)
