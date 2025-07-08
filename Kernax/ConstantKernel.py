import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class

from Kernax import AbstractKernel

@register_pytree_node_class
class ConstantKernel(AbstractKernel):
	def __init__(self, value=1.):
		"""
		Instantiates a constant kernel with the given value.

		:param value: the value of the constant kernel
		"""
		super().__init__(value=value)

	@jit
	def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray = None) -> jnp.ndarray:
		if x2 is None:
			x2 = x1

		if x1.ndim == 1:
			return self.value
		elif x1.ndim == 2:
			return jnp.full((x1.shape[0], x2.shape[0]), self.value)
		elif x1.ndim == 3:
			return jnp.full((x1.shape[0], x1.shape[1], x2.shape[1]), self.value)
		else:
			raise ValueError(f"Unsupported input shape {x1.shape} for ConstantKernel.")