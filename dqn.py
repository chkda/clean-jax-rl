import os

import flax.core
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import gymnax
import distrax

import wandb
from utils.buffer import ExperienceReplay


class QNetwork(nn.Module):
    action_dims: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dims)

        return x


class OffPolicyTrainingState(TrainState):
    target_params: flax.core.FrozenDict

# Below implementation is heavily inspired from CleanRL DQN
def linear_schedule(start_e:float, end_e:float, curr_step:int, duration:int):
    slope = (end_e - start_e)/ duration
    return max(slope * curr_step + start_e, end_e)

