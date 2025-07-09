from jax import jit
from jax.tree_util import register_pytree_node_class
from jax import numpy as jnp

from functools import partial

from Kernax import StaticAbstractKernel, AbstractKernel


class StaticRBFKernel(StaticAbstractKernel):
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
		return kern.variance * jnp.exp(-0.5 * ((x1 - x2) @ (x1 - x2)) / kern.length_scale ** 2)

@register_pytree_node_class
class RBFKernel(AbstractKernel):
	def __init__(self, length_scale=None, variance=None):
		super().__init__(length_scale=length_scale, variance=variance)

		self.static_class = StaticRBFKernel
