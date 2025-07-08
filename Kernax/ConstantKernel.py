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
	def pairwise_cov(self, x1: jnp.ndarray, x2: jnp.ndarray, value=None) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between two vectors.

		:param x1: scalar array
		:param x2: scalar array
		:param value: the value of the constant kernel
		:return: scalar array
		"""
		return value
