from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder
from copy import deepcopy


def update_config(config, **kwargs):
    new_config = deepcopy(config)
    for key, value in kwargs.items():
        if key in config:
            if isinstance(config[key], dict) or isinstance(config[key], ConfigDict):
                new_config[key] = update_config(config[key], **value)
            else:
                new_config[key] = value
        else:
            new_config[key] = value
    return ConfigDict(new_config)


def get_config():
    base_wandb_config = dict(
        project="otter",
        group=placeholder(str),
        entity="otter_rfm",
    )

    model_size, source = "large-patch14", "openai/clip-vit-large-patch14"
    # model_size, source = "base-patch16", "openai/clip-vit-base-patch16"
    # model_size, source = 'base-patch32', 'openai/clip-vit-base-patch32'
    text_processor = "clip_embedding"
    if model_size == "base-patch16":
        text_processor_kwargs = dict(
            source=source,
        )
        pretrained_loaders = ["clip-base16-loader"]
    elif model_size == "large-patch14":
        text_processor_kwargs = dict(
            source=source,
        )
        pretrained_loaders = ["clip-large-loader"]
    elif model_size == "base-patch32":
        text_processor_kwargs = dict(
            source=source,
        )
        pretrained_loaders = ["clip-base32-loader"]
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    base_real_config = dict(
        start_step=0,
        batch_size=64,
        num_steps=int(4e4),
        log_interval=500,
        eval_interval=1000,
        save_interval=1000,
        save_dir=placeholder(str),
        data_path=placeholder(str),
        wandb_resume_id=placeholder(str),
        seed=42,
        text_processor=text_processor,
        text_processor_kwargs=text_processor_kwargs,
        pretrained_loaders=pretrained_loaders,
        wandb=base_wandb_config,
    )

    # params that need to be specified multiple places
    base_data_config = dict(
        dataset_kwargs=[
            dict(
                name="icrt_pickplace",  # 473
                data_dir="dataset/icrt_pickplace/1.0.0",
                shuffle=True,
                action_normalization_mask=[True] * 9 + [False],
                skip_norm=True,
                action_proprio_normalization_type="normal",
                proprio_noise=0.01,
            ),
            dict(
                name="icrt_stack",  # 109
                data_dir="dataset/icrt_stack_mul/1.0.0",
                shuffle=True,
                action_normalization_mask=[True] * 9 + [False],
                skip_norm=True,
                action_proprio_normalization_type="normal",
                proprio_noise=0.01,
            ),
            dict(
                name="icrt_0926",  # 150
                data_dir="dataset/icrt_pickplace_1/1.0.0",
                shuffle=True,
                action_normalization_mask=[True] * 9 + [False],
                skip_norm=True,
                action_proprio_normalization_type="normal",
                proprio_noise=0.01,
            ),
            dict(
                name="icrt_poke",  # 185
                data_dir="dataset/icrt_poke/1.0.0",
                shuffle=True,
                action_normalization_mask=[True] * 9 + [False],
                skip_norm=True,
                action_proprio_normalization_type="normal",
                proprio_noise=0.01,
            ),
            dict(
                name="icrt_drawer",  # 167
                data_dir="dataset/icrt_drawer/1.0.0",
                shuffle=True,
                action_normalization_mask=[True] * 9 + [False],
                skip_norm=True,
                action_proprio_normalization_type="normal",
                proprio_noise=0.01,
            ),
            dict(
                name="icrt_pour",  # 101
                data_dir="dataset/icrt_pour/1.0.0",
                shuffle=True,
                action_normalization_mask=[True] * 9 + [False],
                skip_norm=True,
                action_proprio_normalization_type="normal",
                proprio_noise=0.01,
            ),
        ],
        traj_transform_kwargs=dict(
            goal_relabeling_strategy=None,
            subsample_length=12,
            task_augment_strategy=None,
        ),
        frame_transform_kwargs=dict(
            resize_size=(224, 224),
            image_dropout_prob=0.0,
            image_augment_kwargs=dict(
                primary=dict(
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.1],
                    augment_order=[
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
            shuffle_buffer_size=1600,
            traj_transform_threads=48,
            traj_read_threads=48,
        ),
        sample_weights=[1.5, 1.5, 1.5, 2.5, 4.5, 6],
        prefetch_num_batches=20,
        eval_data=[
            "icrt_pickplace",
            "icrt_poke",
            "icrt_drawer",
            "icrt_pour",
            "icrt_stack",
            "icrt_0926",
        ],
    )

    frozen_keys = None
    base_optimizer_config = dict(
        learning_rate=dict(
            name="cosine",
            init_value=0.0,
            peak_value=3e-4,
            warmup_steps=2000,
            decay_steps=int(4e4),
            end_value=0.0,
        ),
        weight_decay=0.01,
        clip_gradient=1.0,
        frozen_keys=frozen_keys,
        grad_accumulation_steps=None,
    )

    base_model_config = dict(
        policy_kwargs=dict(
            num_layers=8,
            mlp_dim=768,
            num_heads=8,
            dropout_rate=0.1,
            action_pred_horizon=12,
            action_mlp_kwargs=dict(
                n_layers=1,
                hidden_size=1024,
            ),
        ),
    )

    NUM_FUSION_LAYERS = 2
    NUM_READOUTS = 4
    NUM_CAMERAS = 2

    mlp_kwargs = dict(
        output_dim=64,
        n_layers=1,
    )

    proprio_encoder_kwargs = dict(
        mlp_kwargs=mlp_kwargs,
        encode_proprio=True,
    )

    final_config = ConfigDict(
        dict(
            agent="transformer_bc",
            model=update_config(
                base_model_config,
                observation_tokenizers=["proprio-tokenizer"],
                observation_tokenizer_kwargs={
                    "proprio-tokenizer": proprio_encoder_kwargs,
                },
                task_tokenizers=["clear-clip-tokenizer"],
                task_tokenizer_kwargs={
                    "clear-clip-tokenizer": {
                        "source": "openai/clip-vit-large-patch14",
                        "num_cameras": NUM_CAMERAS,
                        "num_readouts": NUM_READOUTS,
                        "num_fusion_layers": NUM_FUSION_LAYERS,
                        "num_text_readouts": 4,
                        "text_ratio": 2,
                        "use_learnable_temperature": True,
                        "add_pe": True,
                        "get_pe": False,
                        "fusion_mlp_dim": 512,
                        "encode_text": True,
                    },
                },
            ),
            optimizer=base_optimizer_config,
            dataset=base_data_config,
            **base_real_config,
        )
    )

    return final_config
