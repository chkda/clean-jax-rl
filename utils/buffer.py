import jax.numpy as jnp
from typing import NamedTuple
from abc import ABC, abstractmethod

# The below implementation is taken in reference to this https://github.com/hamishs/JAX-RL/
class Transition(NamedTuple):
    state: list 
    action: int
    reward: float
    done: bool
    next_state: list
    
    
class Buffer(ABC):
    
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.reset()
        
    @abstractmethod
    def update(self, transition):
        pass
    
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass
        
class ExperienceReplay(Buffer):
    
    def __init__(self, buffer_size, action_dtype=jnp.int32):
        super(ExperienceReplay,self).__init__(buffer_size)
        self.action_dtype = action_dtype
        
    def update(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop()
            
    def __len__(self):
        return len(self.buffer)
    
    def reset(self):
        self.buffer = []
        
    def sample(self, batch_size):
        batch = Transition(*zip(*random.sample(self.buffer, batch_size)))
        
        state = jnp.array(batch.state, dtype=jnp.float32)
        action = jnp.array(batch.action, dtype=self.action_dtype)
        reward = jnp.array(batch.reward, dtype=jnp.float32)
        done = jnp.array(batch.done, dtype=jnp.float32)
        next_state = jnp.array(batch.next_state, dtype=jnp.float32)
        
        return state, action, reward, done, next_state
        
    