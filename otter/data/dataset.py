from functools import partial
import json
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union, List

from absl import logging
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from otter.data import obs_transforms, traj_transforms
from otter.data.utils import goal_relabeling, task_augmentation
from otter.data.utils.data_utils import (
    allocate_threads,
    get_dataset_statistics,
    NormalizationType,
    normalize_action_and_proprio,
    pprint_data_mixture,
    sample_match_keys_uniform,
    tree_map,
    add_proprio_noise,
)
from otter.data.utils.utils import euler_XYZ_to_matrix
from otter.utils.spec import ModuleSpec
import mediapy as media

import time
import random


def icrt_restructure(traj, action_horizon=16, dataset_name="icrt"):
    if "steps" in traj.keys():
        traj = traj["steps"]

    traj_len = tf.shape(traj["observation"]["proprio"])[0]

    gripper_distance = traj["action"][:, -1:]
    action = traj["action"][:, :-1]  # traj_len, 6
    action_rotmat = euler_XYZ_to_matrix(action[:, 3:])  # traj_len, 3, 3
    # make it a transformation matrix of 4*4
    action_t = tf.concat(
        [action_rotmat, action[:, :3, None]], axis=-1
    )  # traj_len, 3, 4
    last_row = tf.eye(4, batch_shape=[traj_len], dtype=action_t.dtype)[:, -1:]
    action_t = tf.concat([action_t, last_row], axis=1)  # traj_len, 4, 4

    proprio = traj["observation"]["proprio"][:, :-1]  # traj_len, 6
    proprio_rotmat = euler_XYZ_to_matrix(proprio[:, 3:])
    proprio_rot6d = tf.reshape(proprio_rotmat[:, :2, :], [-1, 6])
    proprio = tf.concat(
        [proprio[:, :3], proprio_rot6d, traj["observation"]["proprio"][:, -1:]], axis=-1
    )  # traj_len, 9

    state_t = tf.concat(
        [proprio_rotmat, proprio[:, :3, None]], axis=-1
    )  # traj_len, 3, 4
    last_row = tf.eye(4, batch_shape=[traj_len], dtype=action_t.dtype)[:, -1:]
    state_t = tf.concat([state_t, last_row], axis=1)  # traj_len, 4, 4

    action_chunk_indices = tf.range(traj_len)[:, None] + tf.range(
        0, action_horizon
    )  # [traj_len, action_horizon], start from 0
    # repeat the last action at the end of the trajectory rather than going out of bounds
    action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
    # gather
    # for delta action, we need to recalculate the delta action wrt the current proprio
    chunked_action = tf.gather(
        action_t, action_chunk_indices
    )  # [traj_len, action_horizon, proprio_dim]
    broad_casted_proprio = tf.broadcast_to(
        tf.linalg.inv(state_t)[:, None], [traj_len, action_horizon, 4, 4]
    )
    delta_hand_pose = broad_casted_proprio @ chunked_action
    chunked_gripper_distance = tf.gather(gripper_distance, action_chunk_indices)
    delta_rot6d = tf.reshape(
        delta_hand_pose[..., :2, :3], [traj_len, action_horizon, 6]
    )
    delta_hand_action = tf.concat(
        [delta_hand_pose[..., :3, 3], delta_rot6d, chunked_gripper_distance], axis=-1
    )
    action = delta_hand_action  # [traj_len, action_horizon, action_dim]

    prev_action = tf.concat([tf.zeros((1, action.shape[-1])), action[:-1, 0]], axis=0)

    dummy_extrinsics = tf.eye(4, batch_shape=[traj_len])

    if "intrinsics" not in traj.keys():
        traj["intrinsics"] = tf.eye(3, batch_shape=[traj_len])

    new_traj = {
        "observation": {
            "image_primary": traj["observation"][
                "image_primary"
            ],  # traj_len, 224, 224, 3, to be reshaped
            "image_wrist": traj["observation"][
                "image_wrist"
            ],  # traj_len, 224, 224, 3, to be reshapeh
            "proprio": proprio,  # traj_len, 10
            "prev_action": prev_action,  # traj_len, 10
        },
        "state_t": state_t,  # traj_len, 4, 4
        "extrinsics": dummy_extrinsics,  # traj_len, 4, 4
        "intrinsics": traj["intrinsics"],  # traj_len, 3, 3
        "task": {"language_instruction": traj["task"]["language_instruction"]},
        "action": tf.cast(
            action, tf.float32
        ),  # traj_len, 10, last action 0, delta action
        # "raw_action_t": tf.cast(action_t, tf.float32), # traj_len, 4, 4
        "dataset_name": tf.repeat("icrt", traj_len),
    }

    return new_traj


def apply_trajectory_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    subsample_length: Optional[int] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    task_augment_strategy: Optional[str] = None,
    task_augment_kwargs: dict = {},
    max_action_dim: Optional[int] = None,
    max_proprio_dim: Optional[int] = None,
    post_chunk_transforms: Sequence[ModuleSpec] = (),
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Applies common transforms that happen at a trajectory level. Such transforms are usually some sort of
    "relabeling" (e.g. filtering, chunking, adding goals, dropping keys). Transforms that happen in this
    function should have the following properties:

    - They require access to an entire trajectory (i.e. they cannot be applied in a frame-wise manner).
    - They are generally not CPU-intensive, mostly involving moving and copying data.
    - They do not require decoded images.

    Args:
        dataset (dl.DLataset): The dataset to transform.
        train (bool): Whether the dataset is for training (affects subsampling).
        goal_relabeling_strategy (str, optional): The goal relabeling strategy to use, or None for
            no goal relabeling. See `goal_relabeling.py`.
        goal_relabeling_kwargs (dict, optional): Additional keyword arguments to pass to the goal relabeling function.
        window_size (int, optional): The window size to chunk both observations and actions into.
        action_horizon (int, optional): The size of the action chunk (present and future actions) to include in
            the chunked actions.
        subsample_length (int, optional): If provided, trajectories longer than this will be subsampled to
            this length (after goal relabeling and chunking).
        skip_unlabeled (bool, optional): Whether to skip trajectories with no language labels.
        max_action: (float, optional): If provided, trajectories in which *any* action dimension
            of *any* transition has an absolute value larger than this will be skipped.
        max_proprio: (float, optional): If provided, trajectories in which *any* proprio dimension
            of *any* transition has an absolute value larger than this will be skipped.
        task_augment_strategy (str, optional): The task augmentation strategy to use, or None for no task
            augmentation. See `task_augmentation.py`.
        task_augment_kwargs (dict, optional): Additional keyword arguments to pass to the task augmentation
            function.
        max_action_dim (int, optional): If provided, datasets with an action dimension less than this will be
            padded to this dimension.
        max_proprio_dim (int, optional): If provided, datasets with a proprio dimension less than this will be
            padded to this dimension.
        post_chunk_transforms (Sequence[ModuleSpec]): ModuleSpecs of trajectory transforms applied after
            chunking.
        num_parallel_calls (int, optional): number of parallel calls for map operations. Default to AUTOTUNE.
    """
    if skip_unlabeled:
        if "language_instruction" not in dataset.element_spec["task"]:
            raise ValueError(
                "skip_unlabeled=True but dataset does not have language labels."
            )
        dataset = dataset.filter(
            lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != "")
        )

    if max_action is not None:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(tf.math.abs(x["action"]) <= max_action)
        )

    if max_proprio is not None and "proprio" in dataset.element_spec["observation"]:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(
                tf.math.abs(x["observation"]["proprio"]) <= max_proprio
            )
        )

    # marks which entires of the observation and task dicts are padding
    dataset = dataset.traj_map(traj_transforms.add_pad_mask_dict, num_parallel_calls)

    # optionally pads actions and proprio to a consistent number of dimensions
    dataset = dataset.traj_map(
        partial(
            traj_transforms.pad_actions_and_proprio,
            max_action_dim=max_action_dim,
            max_proprio_dim=max_proprio_dim,
        ),
        num_parallel_calls,
    )

    # updates the "task" dict
    if goal_relabeling_strategy is not None:
        dataset = dataset.traj_map(
            partial(
                getattr(goal_relabeling, goal_relabeling_strategy),
                **goal_relabeling_kwargs,
            ),
            num_parallel_calls,
        )

    # must run task augmentation before chunking, in case it changes goal timesteps
    if train and task_augment_strategy is not None:
        # perform task augmentation (e.g., dropping keys)
        dataset = dataset.traj_map(
            partial(
                getattr(task_augmentation, task_augment_strategy),
                **task_augment_kwargs,
            ),
            num_parallel_calls,
        )

    # chunks observations and actions
    # dataset = dataset.traj_map(
    #     partial(
    #         # traj_transforms.chunk_act_obs,
    #         traj_transforms.chunk_act_horizon,
    #         # window_size=window_size,
    #         action_horizon=action_horizon,
    #     ),
    #     num_parallel_calls,
    # )

    # if train and subsample_length is not None:
    #     dataset = dataset.traj_map(
    #         partial(traj_transforms.subsample, subsample_length=subsample_length),
    #         num_parallel_calls,
    #     )
    if subsample_length is not None:
        dataset = dataset.traj_map(
            partial(traj_transforms.subsample, subsample_length=subsample_length),
            num_parallel_calls,
        )

    for transform_spec in post_chunk_transforms:
        dataset = dataset.traj_map(
            ModuleSpec.instantiate(transform_spec),
            num_parallel_calls,
        )

    return dataset


def apply_frame_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    image_augment_kwargs: Union[dict, Mapping[str, dict]] = {},
    resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    depth_resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    image_dropout_prob: float = 0.0,
    image_dropout_keep_key: Optional[str] = None,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Applies common transforms that happen at a frame level. These transforms are usually more
    CPU-intensive, (e.g. decoding or resizing images).

    Args:
        train (bool): Whether the dataset is for training (affects image augmentation).
        dataset (dl.DLataset): The dataset to transform.
        image_augment_kwargs (dict|Mapping[str, dict]): Keyword arguments to pass to the image augmentation
            function. See `dlimp.transforms.augment_image` for documentation of these kwargs. If a dict of
            dicts is provided, then key "k" will be used for "image_{k}" (names determined by `image_obs_keys`
            in `make_dataset_from_rlds`). Augmentation will be skipped for missing keys (so pass an empty dict
            to skip augmentation for all images).
        resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): If provided, images will be resized to
            this size. If a dict of tuples is provided, then key "k" will be used for "image_{k}" (names
            determined by `image_obs_keys` in `make_dataset_from_rlds`). Resizing will be skipped for missing
            keys (so pass an empty dict to skip resizing for all images).
        depth_resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): Same as resize_size, but for depth
            images.
        image_dropout_prob (float): Probability of dropping out images, applied to each image key
            independently. At least one image will always be present.
        image_dropout_keep_key (str, optional): Optionally provide a key to always keep during image dropout
            for example for image observations that are essential for action prediction.
        num_parallel_calls (int): number of parallel calls for frame_map operations. Default to AUTOTUNE.
    """

    # convenience wrapper that takes a function that operates on a non-chunked "observation" dict and applies
    # it to the chunked "observation" dict as well as the non-chunked "task" dict
    def apply_obs_transform(fn: Callable[[dict], dict], frame: dict) -> dict:
        # task is not chunked -- apply fn directly
        frame["task"] = fn(frame["task"])
        # observation is chunked -- apply fn along first axis
        # frame["observation"] = dl.vmap(fn)(frame["observation"])
        frame["observation"] = fn(frame["observation"])
        return frame

    # decode + resize images (and depth images)

    dataset = dataset.frame_map(
        partial(
            apply_obs_transform,
            partial(
                obs_transforms.decode_and_resize,
                resize_size=resize_size,
                depth_resize_size=depth_resize_size,
            ),
        ),
        num_parallel_calls,
    )

    if train:
        # augment all images with the same seed, skipping padding images
        def aug_and_dropout(frame: dict):
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            dropout_fn = partial(
                obs_transforms.image_dropout,
                seed=seed,
                dropout_prob=image_dropout_prob,
                always_keep_key=image_dropout_keep_key,
            )
            aug_fn = partial(
                obs_transforms.augment, seed=seed, augment_kwargs=image_augment_kwargs
            )
            frame = apply_obs_transform(dropout_fn, frame)
            frame = apply_obs_transform(aug_fn, frame)

            return frame

        dataset = dataset.frame_map(aug_and_dropout, num_parallel_calls)

    return dataset


def make_dataset_from_rlds(
    name,
    data_dir,
    train: bool,
    restructure: Union[List[Callable], Callable],
    action_horizon: int = 1,
    shuffle: bool = True,
    action_normalization_mask: Optional[Sequence[bool]] = None,
    dataset_statistics: Optional[Union[dict, str]] = None,
    skip_norm: bool = False,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls=tf.data.AUTOTUNE,
    proprio_noise=None,
    train_ratio=95,
    force_recompute_dataset_statistics: bool = False,
):

    def is_nonzero_length(traj):
        return tf.shape(traj["action"])[0] > 0

    # construct the dataset
    builder = tfds.builder_from_directory(builder_dir=data_dir)

    # calculate statistics if there is no statistics provided
    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)

    elif dataset_statistics is None:
        full_dataset = dl.DLataset.from_rlds(builder, split="all", shuffle=False)

        if isinstance(restructure, list):
            full_dataset = full_dataset.traj_map(partial(restructure[0])).filter(
                is_nonzero_length
            )

            full_dataset = full_dataset.traj_map(
                partial(
                    restructure[1],
                    action_horizon=action_horizon,
                    dataset_name=name,
                ),
            ).filter(is_nonzero_length)
        else:
            full_dataset = full_dataset.traj_map(
                partial(restructure, action_horizon=action_horizon, dataset_name=name),
            ).filter(is_nonzero_length)

        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                f"{name}",
                f"_action_horizon{action_horizon}",
            ),
            save_dir=builder.data_dir,
            force_recompute=force_recompute_dataset_statistics,
        )

    dataset_statistics = tree_map(np.array, dataset_statistics)

    if "prev_action" not in dataset_statistics.keys():
        dataset_statistics["prev_action"] = {}
        for k in dataset_statistics["action"].keys():
            dataset_statistics["prev_action"][k] = dataset_statistics["action"][k][0]

    if action_normalization_mask is not None:
        dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)

    if "val" not in builder.info.splits:
        split = f"train[:{train_ratio}%]" if train else "train[95%:]"

    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads
    )

    if isinstance(restructure, list):
        dataset = dataset.traj_map(partial(restructure[0])).filter(is_nonzero_length)

        dataset = dataset.traj_map(
            partial(
                restructure[1],
                action_horizon=action_horizon,
                dataset_name=name,
            ),
        ).filter(is_nonzero_length)
    else:
        dataset = dataset.traj_map(
            partial(restructure, action_horizon=action_horizon, dataset_name=name),
        ).filter(is_nonzero_length)

    if not skip_norm:
        dataset = dataset.map(
            partial(
                normalize_action_and_proprio,
                metadata=dataset_statistics,
                normalization_type=action_proprio_normalization_type,
            ),
            num_parallel_calls=num_parallel_calls,
        )
    else:
        logging.warning(
            "Dataset normalization turned off -- set skip_norm=False to apply normalization."
        )

    if proprio_noise is not None:
        dataset = dataset.map(
            partial(
                add_proprio_noise,
                noise=proprio_noise,
            ),
            num_parallel_calls=num_parallel_calls,
        )

    return dataset, dataset_statistics


def make_single_dataset(
    dataset_kwargs: dict,
    *,
    train: bool,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
    batch_size=None,
) -> dl.DLataset:
    """Creates a single dataset from kwargs. Returns a dataset of trajectories.

    Args:
        dataset_kwargs: kwargs passed to `make_dataset_from_rlds` that are dataset-specific.
        train: whether this is a training or validation dataset.
        traj_transform_kwargs: kwargs passed to 'apply_trajectory_transforms'.
        frame_transform_kwargs: kwargs passed to 'get_frame_transforms'.
    """
    dataset, dataset_statistics = make_dataset_from_rlds(
        **dataset_kwargs,
        train=train,
    )
    dataset = apply_trajectory_transforms(
        dataset.repeat(), **traj_transform_kwargs, train=train
    )
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    # this seems to reduce memory usage without affecting speed
    # dataset = dataset.with_ram_budget(1)

    dataset = dataset.ignore_errors(log_warning=False)
    # save for later
    dataset.dataset_statistics = dataset_statistics
    return dataset


def make_interleaved_dataset(
    dataset_kwargs_list: Sequence[dict],
    sample_weights: Optional[Sequence[float]] = None,
    *,
    train: bool,
    shuffle_buffer_size: int,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
    batch_size: Optional[int] = None,
    balance_weights: bool = False,
    traj_transform_threads: Optional[int] = None,
    traj_read_threads: Optional[int] = None,
) -> dl.DLataset:
    """Creates an interleaved dataset from list of dataset kwargs. Returns a dataset of batched frames.

    Args:
        dataset_kwargs_list: list of kwargs, each element of which is passed to `make_dataset_from_rlds`.
            "num_parallel_calls" and "num_parallel_reads" are overidden using `traj_transform_threads` and
            `traj_read_threads`, respectively.
        sample_weights: sampling weights for each dataset in list. If None, defaults to uniform.
        train: whether this is a training or validation dataset.
        shuffle_buffer_size: size of the dataset shuffle buffer (in number of frames).
        traj_transform_kwargs: kwargs passed to `apply_trajectory_transforms`. "num_parallel_calls" is
            overidden using `traj_transform_threads`.
        frame_transform_kwargs: kwargs passed to `apply_frame_transforms`.
        batch_size: batch size, if not provided output is not batched.
        balance_weights: if True, the sample weights are multiplied by the number of frames in each dataset.
            This makes it so that, if all the sample weights are equal, one full iteration through the interleaved
            dataset will correspond to one full iteration through each individual dataset (only in expectation,
            since in practice the sampling is random).
        traj_transform_threads: total number of parallel calls for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
        traj_read_threads: total number of parallel read workers for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
    """
    # default to uniform sampling
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(
            f"sample_weights must be None or have length {len(dataset_kwargs_list)}."
        )

    # go through datasets once to get sizes
    dataset_sizes = []
    all_dataset_statistics = {}
    for dataset_kwargs in dataset_kwargs_list:
        _, dataset_statistics = make_dataset_from_rlds(**dataset_kwargs, train=train)
        dataset_sizes.append(dataset_statistics["num_transitions"])
        assert (
            dataset_kwargs["name"] not in all_dataset_statistics
        ), f"Duplicate name {dataset_kwargs['name']}"
        all_dataset_statistics[dataset_kwargs["name"]] = dataset_statistics

    # balance and normalize weights
    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    # allocate threads based on weights
    threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

    logging.info("Threads per dataset: %s", threads_per_dataset)
    logging.info("Reads per dataset: %s", reads_per_dataset)

    # construct datasets
    datasets = []
    for dataset_kwargs, threads, reads in zip(
        dataset_kwargs_list,
        threads_per_dataset,
        reads_per_dataset,
    ):
        if "dataset_statistics" in dataset_kwargs.keys():
            del dataset_kwargs["dataset_statistics"]

        dataset, _ = make_dataset_from_rlds(
            **dataset_kwargs,
            train=train,
            num_parallel_calls=threads,
            num_parallel_reads=reads,
            dataset_statistics=all_dataset_statistics[dataset_kwargs["name"]],
        )
        dataset = apply_trajectory_transforms(
            dataset.repeat(),
            **traj_transform_kwargs,
            num_parallel_calls=threads,
            train=train,
        )  # .flatten(num_parallel_calls=threads)
        datasets.append(dataset)

    # interleave at the traj level and then shuffle
    dataset: dl.DLataset = dl.DLataset.sample_from_datasets(
        datasets, sample_weights
    ).shuffle(shuffle_buffer_size)

    # apply frame transforms
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # sequential batch (parallel batch seems to use much more memory)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    dataset = dataset.ignore_errors(log_warning=True)

    # save for later
    dataset.dataset_statistics = all_dataset_statistics
    dataset.sample_weights = sample_weights
    return dataset
