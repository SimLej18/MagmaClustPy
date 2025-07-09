import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class

from functools import partial

from Kernax import StaticAbstractKernel, AbstractKernel


class StaticConstantKernel(StaticAbstractKernel):
	@classmethod
	@partial(jit, static_argnums=(0,))
	def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		return kern.value  # The constant value is returned regardless of the inputs


@register_pytree_node_class
class ConstantKernel(AbstractKernel):
	def __init__(self, value=1.):
		"""
		Instantiates a constant kernel with the given value.

		:param value: the value of the constant kernel
		"""
		super().__init__(value=value)
		self.static_class = StaticConstantKernel
