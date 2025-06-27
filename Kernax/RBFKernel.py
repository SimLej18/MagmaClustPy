from jax import jit
from jax.tree_util import register_pytree_node_class
from jax import numpy as jnp

from Kernax import AbstractKernel


@register_pytree_node_class
class RBFKernel(AbstractKernel):
	def __init__(self, length_scale=None, variance=None, **kwargs):
		if length_scale is None:
			length_scale = jnp.array([1.])
		if variance is None:
			variance = jnp.array([1.])
		super().__init__(length_scale=length_scale, variance=variance, **kwargs)

	@jit
	def compute_scalar(self, x1: jnp.ndarray, x2: jnp.ndarray, length_scale=None, variance=None) -> jnp.ndarray:
		return variance * jnp.exp(-0.5 * (x1 - x2) ** 2 / length_scale ** 2)
