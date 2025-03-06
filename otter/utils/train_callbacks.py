from dataclasses import dataclass
from functools import partial
import logging
import os
from typing import Callable, Mapping, Optional

import flax
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import tensorflow as tf
import tqdm

from otter.data.dataset import make_single_dataset
from otter.data.utils.text_processing import TextProcessor
from otter.utils.train_utils import TrainState


class Callback:
    def __call__(self, train_state: TrainState, step: int):
        raise NotImplementedError


def create_validation_dataset(
    dataset_kwargs: dict,
    traj_transform_kwargs: dict,
    frame_transform_kwargs: dict,
    train: bool = False,
):
    """Creates a dataset for validation and visualization purposes.

    Takes the training configuration and overwrites default parameters with more conservative
    options to ensure stable memory consumption.
    """
    return make_single_dataset(
        dataset_kwargs={
            **dataset_kwargs,
            "num_parallel_reads": 4,
            "num_parallel_calls": 4,
            "shuffle": False,
        },
        traj_transform_kwargs={
            **traj_transform_kwargs,
            "num_parallel_calls": 4,
        },
        frame_transform_kwargs={
            **frame_transform_kwargs,
            "num_parallel_calls": 16,
        },
        train=train,
    )


@dataclass
class SaveCallback(Callback):
    """Callback that saves checkpoints to `save_dir`. If `save_dir` is None, does nothing."""

    save_dir: Optional[str]

    def __post_init__(self):
        if self.save_dir is not None:
            if not self.save_dir.startswith("gs://"):
                self.save_dir = os.path.abspath(self.save_dir)
            if jax.process_index() == 0:
                tf.io.gfile.makedirs(self.save_dir)
                logging.info(f"Created {self.save_dir}")
            # make checkpointers
            # only keep latest full TrainState
            self.state_checkpointer = orbax.checkpoint.CheckpointManager(
                tf.io.gfile.join(self.save_dir, "state"),
                orbax.checkpoint.PyTreeCheckpointer(),
                options=orbax.checkpoint.CheckpointManagerOptions(
                    max_to_keep=1,
                ),
            )
            # keep every params checkpoint
            self.params_checkpointer = orbax.checkpoint.CheckpointManager(
                self.save_dir,
                orbax.checkpoint.PyTreeCheckpointer(),
            )

    def __call__(self, train_state: TrainState, step: int):
        if self.save_dir is not None:
            train_state.model.save_pretrained(
                step, checkpoint_manager=self.params_checkpointer
            )
            self.state_checkpointer.save(
                step,
                train_state,
                {"save_args": orbax_utils.save_args_from_target(train_state)},
            )
