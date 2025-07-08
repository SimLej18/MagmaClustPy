import jax.numpy as jnp
from jax import jit, vmap
from jax.tree_util import register_pytree_node_class
from jax.lax import cond

from Kernax.OperatorKernels import SumKernel, ProductKernel
from Kernax.WrapperKernels import NegKernel


@register_pytree_node_class
class AbstractKernel:
	def __init__(self, **kwargs):
		"""
		Instatiates a kernel with the given hyperparameters.
		https://docs.jax.dev/en/latest/pytrees.html#custom-pytrees-and-initialization
		:param kwargs: the hyperparameters of the kernel, as keyword arguments.
		"""
		# Check that hyperparameters are all jnp arrays/scalars or kernels
		for key, value in kwargs.items():
			# If given value is numeric, convert it to a jnp array
			if isinstance(value, (int, float)):
				kwargs[key] = jnp.array(float(value))

		# Register hyperparameters in *kwargs* as instance attributes
		self.__dict__.update(kwargs)

	def __str__(self):
		return f"{self.__class__.__name__}({', '.join([f'{key}={value}' for key, value in self.__dict__.items()])})"

	def __repr__(self):
		return str(self)

	@jit
	def check_kwargs(self, **kwargs):
		for key in self.__dict__:
			if key not in kwargs:
				kwargs[key] = self.__dict__[key]
		return kwargs

	@jit
	def __call__(self, x1, x2=None, **kwargs):
		# If no x2 is provided, we compute the covariance between x1 and itself
		if x2 is None:
			x2 = x1

		# Turn scalar inputs into vectors
		x1, x2 = jnp.atleast_2d(x1), jnp.atleast_2d(x2)

		# Check kwargs
		kwargs = self.check_kwargs(**kwargs)

		# Call the appropriate method
		if jnp.ndim(x1) == 1 and jnp.ndim(x2) == 1:
			return self.pairwise_cov_if_not_nan(x1, x2, **kwargs)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 1:
			return self.cross_cov_vector_if_not_nan(x1, x2, **kwargs)
		elif jnp.ndim(x1) == 1 and jnp.ndim(x2) == 2:
			return self.cross_cov_vector_if_not_nan(x2, x1, **kwargs)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 2:
			return self.cross_cov_matrix(x1, x2, **kwargs)
		elif jnp.ndim(x1) == 3 and jnp.ndim(x2) == 3:
			return self.cross_cov_batch(x1, x2, **kwargs)
		else:
			return jnp.nan

	# Methods to use Kernel as a PyTree
	def tree_flatten(self):
		return tuple(self.__dict__.values()), None  # No static values

	@classmethod
	def tree_unflatten(cls, _, children):
		# This class being abstract, this function fails when called on an "abstract instance",
		# as we don't know the number of parameters the constructor expects, yet we send it children.
		# On a subclass, this will work as expected as long as the constructor has a clear number of
		# kwargs as parameters.
		return cls(*children)

	@jit
	def pairwise_cov_if_not_nan(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Returns NaN if either x1 or x2 is NaN, otherwise calls the compute_scalar method.

		:param x1: scalar array
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: scalar array
		"""
		return cond(jnp.any(jnp.isnan(x1) | jnp.isnan(x2)), lambda _: jnp.nan,
		            lambda _: self.pairwise_cov(x1, x2, **kwargs), None)

	@jit
	def pairwise_cov(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between two vectors.

		:param x1: scalar array
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: scalar array
		"""
		return jnp.array(jnp.nan)  # To be overwritten in subclasses

	@jit
	def cross_cov_vector(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the kernel cross covariance values between an array of vectors (matrix) and a vector.

		:param x1: vector array (N, )
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: vector array (N, )
		"""
		return vmap(lambda x: self.pairwise_cov_if_not_nan(x, x2, **kwargs), in_axes=0)(x1)

	@jit
	def cross_cov_vector_if_not_nan(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Returns an array of NaN if scalar is NaN, otherwise calls the compute_vector method.

		:param x1: vector array (N, )
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: vector array (N, )
		"""
		return cond(jnp.any(jnp.isnan(x2)), lambda _: jnp.full(len(x1), jnp.nan), lambda _: self.cross_cov_vector(x1, x2, **kwargs),
		            None)

	@jit
	def cross_cov_matrix(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the kernel covariance matrix between two vector arrays.

		:param x1: vector array (N, )
		:param x2: vector array (M, )
		:param kwargs: hyperparameters of the kernel
		:return: matrix array (N, M)
		"""
		return vmap(lambda x: self.cross_cov_vector_if_not_nan(x2, x, **kwargs), in_axes=0)(x1)

	@jit
	def cross_cov_batch(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the kernel covariance matrix between two batched vector arrays.

		:param x1: vector array (B, N)
		:param x2: vector array (B, M)
		:param kwargs: hyperparameters of the kernel. Each HP that is a scalar will be shared to the whole batch, and
		each HP that is a vector will be distinct and thus must have shape (B, )
		:return: tensor array (B, N, M)
		"""
		# vmap(self.compute_matrix)(x1, x2, **kwargs)
		shared_hps = {key: value for key, value in kwargs.items() if jnp.isscalar(value)}
		distinct_hps = {key: value for key, value in kwargs.items() if not jnp.isscalar(value)}

		return vmap(lambda x, y, hps: self.cross_cov_matrix(x, y, **hps, **shared_hps), in_axes=(0, 0, 0))(x1, x2, distinct_hps)

	def __add__(self, other):
		return SumKernel(self, other)

	def __radd__(self, other):
		return SumKernel(other, self)

	def __neg__(self):
		return NegKernel(self)

	def __mul__(self, other):
		return ProductKernel(self, other)

	def __rmul__(self, other):
		return ProductKernel(other, self)
