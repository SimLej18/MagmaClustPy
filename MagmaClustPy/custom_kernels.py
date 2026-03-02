from jax import numpy as jnp
from jax import Array
import equinox as eqx
from equinox import filter_jit

from kernax import StaticAbstractKernel, AbstractKernel
from kernax.transforms import to_constrained, to_unconstrained


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
		kern = eqx.combine(kern)
		return kern.variance * jnp.exp(-0.5 * ((x1 - x2) @ (x1 - x2)) / kern.length_scale ** 2)

class RBFKernel(AbstractKernel):
	"""
	RBF (Radial Basis Function) Kernel with constrained positive parameters.

	Both length_scale and variance are constrained to be positive.
	"""
	static_class = StaticRBFKernel

	_unconstrained_length_scale: Array = eqx.field(converter=jnp.asarray)
	_unconstrained_variance: Array = eqx.field(converter=jnp.asarray)

	def __init__(self, length_scale, variance, **kwargs):
		"""
		Initialize the RBF kernel.

		:param length_scale: length scale parameter (ℓ) - must be positive
		:param variance: variance parameter (σ²) - must be positive
		"""
		# Validate parameters are positive
		length_scale = jnp.array(length_scale)
		variance = jnp.array(variance)
		length_scale = eqx.error_if(length_scale, jnp.any(length_scale <= 0), "length_scale must be positive.")
		variance = eqx.error_if(variance, jnp.any(variance <= 0), "variance must be positive.")

		# Initialize parent
		super().__init__(**kwargs)

		# Store in unconstrained space
		self._unconstrained_length_scale = to_unconstrained(jnp.asarray(length_scale))
		self._unconstrained_variance = to_unconstrained(jnp.asarray(variance))

	@property
	def length_scale(self) -> Array:
		"""Get length_scale in constrained (positive) space."""
		return to_constrained(self._unconstrained_length_scale)

	@property
	def variance(self) -> Array:
		"""Get variance in constrained (positive) space."""
		return to_constrained(self._unconstrained_variance)


class StaticSEMagmaKernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between two vectors.

		Note: This kernel uses log-space parameterization in its formula.
		The stored parameters are still constrained to be positive.

		:param kern: the kernel to use, containing hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		kern = eqx.combine(kern)
		return jnp.exp(jnp.log(kern.variance) - (1.0 / kern.length_scale) * jnp.sum((x1 - x2) ** 2) * 0.5)

class SEMagmaKernel(AbstractKernel):
	"""
	Squared Exponential Magma Kernel with constrained positive parameters.

	This kernel uses a log-space parameterization internally but stores
	parameters in their natural positive-constrained space.
	"""
	static_class = StaticSEMagmaKernel

	_unconstrained_length_scale: Array = eqx.field(converter=jnp.asarray)
	_unconstrained_variance: Array = eqx.field(converter=jnp.asarray)

	def __init__(self, length_scale, variance, **kwargs):
		"""
		Initialize the SE Magma kernel.

		:param length_scale: length scale parameter (ℓ) - must be positive
		:param variance: variance parameter (σ²) - must be positive
		"""
		# Validate parameters are positive
		length_scale = jnp.array(length_scale)
		variance = jnp.array(variance)
		length_scale = eqx.error_if(length_scale, jnp.any(length_scale <= 0), "length_scale must be positive.")
		variance = eqx.error_if(variance, jnp.any(variance <= 0), "variance must be positive.")

		# Initialize parent
		super().__init__(**kwargs)

		# Store in unconstrained space
		self._unconstrained_length_scale = to_unconstrained(jnp.asarray(length_scale))
		self._unconstrained_variance = to_unconstrained(jnp.asarray(variance))

	@property
	def length_scale(self) -> Array:
		"""Get length_scale in constrained (positive) space."""
		return to_constrained(self._unconstrained_length_scale)

	@property
	def variance(self) -> Array:
		"""Get variance in constrained (positive) space."""
		return to_constrained(self._unconstrained_variance)


class StaticFeatureKernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between two vectors.

		:param kern: kernel instance containing the hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		kern = eqx.combine(kern)

		# As the formula only involves diagonal matrices, we can compute directly with vectors
		sigma_diag = kern.length_scale_1 + kern.length_scale_2 + kern.length_scale_u  # Σ
		sigma_det = jnp.prod(sigma_diag)  # |Σ|
		diff = x1 - x2  # x - x'

		# Compute the quadratic form: (x - x')^T Sigma^{-1} (x - x')
		# Since Sigma^{-1} is diagonal, this simplifies to sum of (diff_i^2 * sigma_inv_diag_i)
		quadratic_form = jnp.sum(diff**2 / sigma_diag)

		return kern.variance_1 * kern.variance_2 /(((2 * jnp.pi)**(len(x1)/2)) * jnp.sqrt(sigma_det)) * jnp.exp(-0.5 * quadratic_form)


class FeatureKernel(AbstractKernel):
	"""
	Feature Kernel with multiple positive-constrained length scales and variances.

	All parameters (length_scale_1, length_scale_2, length_scale_u, variance_1, variance_2)
	are constrained to be positive.
	"""

	_unconstrained_length_scale_1: Array = eqx.field(converter=jnp.asarray)
	_unconstrained_length_scale_2: Array = eqx.field(converter=jnp.asarray)
	_unconstrained_length_scale_u: Array = eqx.field(converter=jnp.asarray)
	_unconstrained_variance_1: Array = eqx.field(converter=jnp.asarray)
	_unconstrained_variance_2: Array = eqx.field(converter=jnp.asarray)

	static_class = StaticFeatureKernel

	def __init__(self, length_scale_1, length_scale_2, length_scale_u, variance_1, variance_2, **kwargs):
		"""
		Initialize the Feature kernel.

		:param length_scale_1: first length scale parameter - must be positive
		:param length_scale_2: second length scale parameter - must be positive
		:param length_scale_u: uncertainty length scale parameter - must be positive
		:param variance_1: first variance parameter - must be positive
		:param variance_2: second variance parameter - must be positive
		"""
		# Validate all parameters are positive
		length_scale_1 = jnp.array(length_scale_1)
		length_scale_2 = jnp.array(length_scale_2)
		length_scale_u = jnp.array(length_scale_u)
		variance_1 = jnp.array(variance_1)
		variance_2 = jnp.array(variance_2)

		length_scale_1 = eqx.error_if(length_scale_1, jnp.any(length_scale_1 <= 0), "length_scale_1 must be positive.")
		length_scale_2 = eqx.error_if(length_scale_2, jnp.any(length_scale_2 <= 0), "length_scale_2 must be positive.")
		length_scale_u = eqx.error_if(length_scale_u, jnp.any(length_scale_u <= 0), "length_scale_u must be positive.")
		variance_1 = eqx.error_if(variance_1, jnp.any(variance_1 <= 0), "variance_1 must be positive.")
		variance_2 = eqx.error_if(variance_2, jnp.any(variance_2 <= 0), "variance_2 must be positive.")

		# Initialize parent
		super().__init__(**kwargs)

		# Store in unconstrained space
		self._unconstrained_length_scale_1 = to_unconstrained(jnp.asarray(length_scale_1))
		self._unconstrained_length_scale_2 = to_unconstrained(jnp.asarray(length_scale_2))
		self._unconstrained_length_scale_u = to_unconstrained(jnp.asarray(length_scale_u))
		self._unconstrained_variance_1 = to_unconstrained(jnp.asarray(variance_1))
		self._unconstrained_variance_2 = to_unconstrained(jnp.asarray(variance_2))

	@property
	def length_scale_1(self) -> Array:
		"""Get length_scale_1 in constrained (positive) space."""
		return to_constrained(self._unconstrained_length_scale_1)

	@property
	def length_scale_2(self) -> Array:
		"""Get length_scale_2 in constrained (positive) space."""
		return to_constrained(self._unconstrained_length_scale_2)

	@property
	def length_scale_u(self) -> Array:
		"""Get length_scale_u in constrained (positive) space."""
		return to_constrained(self._unconstrained_length_scale_u)

	@property
	def variance_1(self) -> Array:
		"""Get variance_1 in constrained (positive) space."""
		return to_constrained(self._unconstrained_variance_1)

	@property
	def variance_2(self) -> Array:
		"""Get variance_2 in constrained (positive) space."""
		return to_constrained(self._unconstrained_variance_2)


class StaticMOKernel(StaticAbstractKernel):
	@classmethod
	@filter_jit
	def pairwise_cov(cls, kern: AbstractKernel, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between two vectors.

		:param kern: kernel instance containing the hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		kern = eqx.combine(kern)

		# As the formula only involves diagonal matrices, we can compute directly with vectors
		sigma_diag = jnp.exp(kern.length_scale_1) + jnp.exp(kern.length_scale_2) + jnp.exp(kern.length_scale_u)  # Σ
		sigma_det = jnp.prod(sigma_diag)  # |Σ|
		diff = x1 - x2  # x - x'

		# Compute the quadratic form: (x - x')^T Sigma^{-1} (x - x')
		# Since Sigma^{-1} is diagonal, this simplifies to sum of (diff_i^2 * sigma_inv_diag_i)
		quadratic_form = jnp.sum(diff**2 / sigma_diag)

		return jnp.exp(kern.variance_1) * jnp.exp(kern.variance_2) /(((2 * jnp.pi)**(len(x1)/2)) * jnp.sqrt(sigma_det)) * jnp.exp(-0.5 * quadratic_form)


class MOKernel(AbstractKernel):
	"""
	Squared Exponential (aka "RBF" or "Gaussian") Kernel
	"""

	length_scale_1: Array = eqx.field(converter=jnp.asarray)
	length_scale_2: Array = eqx.field(converter=jnp.asarray)
	length_scale_u: Array = eqx.field(converter=jnp.asarray)
	variance_1: Array = eqx.field(converter=jnp.asarray)
	variance_2: Array = eqx.field(converter=jnp.asarray)

	static_class = StaticMOKernel

	def __init__(self, length_scale_1, length_scale_2, length_scale_u, variance_1, variance_2):
		super().__init__()
		self.length_scale_1 = length_scale_1
		self.length_scale_2 = length_scale_2
		self.length_scale_u = length_scale_u
		self.variance_1 = variance_1
		self.variance_2 = variance_2