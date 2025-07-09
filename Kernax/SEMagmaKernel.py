from jax import jit
from jax.tree_util import register_pytree_node_class
from jax import numpy as jnp

from functools import partial

from Kernax import StaticAbstractKernel, AbstractKernel


class StaticSEMagmaKernel(StaticAbstractKernel):
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
		return jnp.exp(kern.variance - jnp.exp(-kern.length_scale) * jnp.sum((x1 - x2) ** 2) * 0.5)

@register_pytree_node_class
class SEMagmaKernel(AbstractKernel):
	def __init__(self, length_scale=None, variance=None, **kwargs):
		super().__init__(length_scale=length_scale, variance=variance, **kwargs)
		self.static_class = StaticSEMagmaKernel
