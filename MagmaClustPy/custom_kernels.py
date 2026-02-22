from jax import numpy as jnp
from jax import Array
import equinox as eqx
from equinox import filter_jit

from kernax import StaticAbstractKernel, AbstractKernel


class StaticRBFKernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		return kern.variance * jnp.exp(-0.5 * ((x1 - x2) @ (x1 - x2)) / kern.length_scale ** 2)

class RBFKernel(AbstractKernel):
	static_class = StaticRBFKernel

	length_scale: Array = eqx.field(converter=jnp.asarray)
	variance: Array = eqx.field(converter=jnp.asarray)

	def __init__(self, length_scale, variance):
		"""
		:param length_scale: length scale parameter (ℓ)
		:param variance: variance parameter (σ²)
		"""
		super().__init__()
		self.length_scale = length_scale
		self.variance = variance


class StaticSEMagmaKernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		return jnp.exp(kern.variance - jnp.exp(-kern.length_scale) * jnp.sum((x1 - x2) ** 2) * 0.5)

class SEMagmaKernel(AbstractKernel):
	static_class = StaticSEMagmaKernel

	length_scale: Array = eqx.field(converter=jnp.asarray)
	variance: Array = eqx.field(converter=jnp.asarray)

	def __init__(self, length_scale, variance):
		"""
		:param length_scale: length scale parameter (ℓ)
		:param variance: variance parameter (σ²)
		"""
		super().__init__()
		self.length_scale = length_scale
		self.variance = variance
