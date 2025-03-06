# adapted from https://github.com/google-research/robotics_transformer/blob/master/transformer_network.py
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from otter.model.transformer import Transformer
from otter.typing import PRNGKey, Sequence
from otter.model.tokenizers import MLP
import logging
from otter.model.tokenizers import tokenizers


class ActionHead(nn.Module):
    action_mlp_kwargs: dict
    action_pred_horizon: int
    action_dim: int

    def setup(self):
        self.action_head = MLP(
            output_dim=self.action_pred_horizon * self.action_dim,
            **self.action_mlp_kwargs,
        )

    def __call__(self, x):
        return self.action_head(x)


class TransformerPolicy(nn.Module):
    """
    Transformer that models trajectories.
    """

    observation_tokenizers: Sequence[nn.Module]
    task_tokenizers: Sequence[nn.Module]
    action_mlp_kwargs: dict
    num_layers: int = 4
    mlp_dim: int = 1024
    num_heads: int = 8
    dropout_rate: float = 0.1
    time_sequence_length: int = 1
    action_pred_horizon: int = 16
    action_dim: int = 10  # 6 rot + 3 ee + 1 gripper

    def setup(self):
        self.transformer = Transformer(
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        self.action_head = ActionHead(
            action_mlp_kwargs=self.action_mlp_kwargs,
            action_pred_horizon=self.action_pred_horizon,
            action_dim=self.action_dim,
        )
        self.tokens_per_action = self.action_dim
        self.causal_mask = jnp.tril(
            jnp.ones((self.time_sequence_length, self.time_sequence_length))
        )

    def __call__(
        self,
        observations,
        task,
        actions,
        train: bool = False,
    ):
        output = self.transformer_call(
            observations,
            task,
            train=train,
        )
        pred_action = self.action_head(output)
        pred_action = jnp.reshape(
            pred_action,
            (
                pred_action.shape[0],
                pred_action.shape[1],
                self.action_pred_horizon,
                self.action_dim,
            ),
        )
        assert (
            pred_action.shape == actions.shape
        ), f"Expected shape {actions.shape}, got {pred_action.shape}"
        # add l1 loss
        action_loss = jnp.mean(jnp.abs(pred_action - actions))
        return {"loss": action_loss, "pred_action": pred_action}

    def transformer_call(
        self,
        observations,
        task,
        train: bool = False,
    ):
        # get lang, rgb, and proprio tokens
        all_tokens = self.get_tokens(
            observations,
            task,
            train=train,
        )
        assert (
            all_tokens.shape[1] == self.time_sequence_length
        ), f"Expected time sequence length {self.time_sequence_length}, got {all_tokens.shape[1]}"

        # transformer call
        output = self.transformer(
            all_tokens, attention_mask=self.causal_mask, train=train
        )

        return output

    def predict_action(
        self,
        observations,
        task,
        train: bool = False,
        argmax: bool = False,
        rng: PRNGKey = None,
        temperature: float = 1.0,
    ):
        output = self.transformer_call(
            observations,
            task,
            train=train,
        )
        action = self.action_head(output)
        action = jnp.reshape(
            action,
            (
                action.shape[0],
                action.shape[1],
                self.action_pred_horizon,
                self.action_dim,
            ),
        )
        return action

    def get_tokens(self, observations, task, train=True):
        """
        Tokenize obervation/action history and task (either image or language or proprio).
        """
        task_tokens = self.task_tokenizers[0](
            observations, task, train=train
        )  # assume only one task tokenizer
        obs_tokens = self.observation_tokenizers[0](
            observations, task, train=train
        )  # assume only one observation tokenizer
        all_tokens = jnp.concatenate([obs_tokens, task_tokens], axis=-1)
        return all_tokens


class OtterModule(nn.Module):
    """
    Bundles OctoTransformer with various heads (useful for keeping all parameters in one place).
    """

    transformer_policy: TransformerPolicy

    def __call__(self, observations, tasks, actions, train=True, verbose=False):
        transformer_outputs = self.transformer_policy(
            observations, tasks, actions, train=train
        )
        return transformer_outputs

    def predict_action(
        self,
        observations,
        tasks,
        train=True,
        argmax=False,
        rng=None,
        temperature=1.0,
    ):
        return self.transformer_policy.predict_action(
            observations,
            tasks,
            train=train,
            argmax=argmax,
            rng=rng,
            temperature=temperature,
        )

    @classmethod
    def create(
        cls,
        observation_tokenizer_kwargs,
        task_tokenizer_kwargs,
        policy_kwargs,
        **kwargs,
    ):
        if len(kwargs) > 0:
            logging.warn(f"Extra kwargs passed into create_model_def: {kwargs}")
        observation_tokenizer_defs = tuple(
            tokenizers[k](**kwargs)
            for k, kwargs in observation_tokenizer_kwargs.items()
        )
        task_tokenizer_defs = tuple(
            tokenizers[k](**kwargs) for k, kwargs in task_tokenizer_kwargs.items()
        )
        model_def = TransformerPolicy(
            observation_tokenizers=observation_tokenizer_defs,
            task_tokenizers=task_tokenizer_defs,
            **policy_kwargs,
        )
        return cls(
            transformer_policy=model_def,
        )


if __name__ == "__main__":
    import jax
    import numpy as np

    traj_len = 5
    new_traj = {
        "observation": {
            "image": np.zeros((traj_len, 3, 224, 224)),
            "proprio": np.zeros((traj_len, 10)),
        },
        "task": np.zeros((traj_len, 5)),
        "action": np.zeros((traj_len, 10)),  # traj_len, 10, last action 0
    }
