from typing import NamedTuple


# The below implementation is taken in reference to this https://github.com/hamishs/JAX-RL/
class Transition(NamedTuple):
    state: list 
    action: int
    reward: float
    done: bool
    next_state: list