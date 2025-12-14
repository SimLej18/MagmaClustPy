from abc import ABC, abstractmethod
from equinox import filter_jit
from jax import numpy as jnp


class BasePriorMean(ABC):
	@abstractmethod
	def __call__(self, grid):
		raise NotImplementedError

class ZeroMean(BasePriorMean):
	@filter_jit
	def __call__(self, grid):
		return jnp.zeros_like(grid[:, 0])
