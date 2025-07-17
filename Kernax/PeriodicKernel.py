from jax import jit
from jax.tree_util import register_pytree_node_class
from jax import numpy as jnp

from functools import partial

from Kernax import StaticAbstractKernel, AbstractKernel


class StaticPeriodicKernel(StaticAbstractKernel):
	@classmethod
	@partial(jit, static_argnums=(0,))
	def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
        Compute the periodic kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters (length_scale, variance, period).
		:param x1: scalar array
		:param x2: scalar array
		:return: covariance value (scalar)
		"""
		dist = jnp.linalg.norm(x1 - x2)

		return kern.variance * jnp.exp(-2 * jnp.sin(jnp.pi * dist / kern.period)**2 / kern.length_scale**2)

@register_pytree_node_class
class PeriodicKernel(AbstractKernel):
	def __init__(self, length_scale=None, variance=None, period=None):
		"""
		:param length_scale: length scale parameter (ℓ)
		:param variance: variance parameter (σ²)
		:param period: period parameter (p)
		"""
		super().__init__(length_scale=length_scale, variance=variance, period=period)

		self.static_class = StaticPeriodicKernel