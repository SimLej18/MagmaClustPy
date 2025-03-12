from jax import jit, vmap, lax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class AbstractKernel:
	def __init__(self, **kwargs):
		# Check that hyperparameters are all jnp arrays/scalars
		for key, value in kwargs.items():
			if not isinstance(value, jnp.ndarray):  # Check type
				raise ValueError(f"Parameter {key} must be a jnp.ndarray.")
			else:  # Check dimensionality
				if len(value.shape) > 1:
					raise ValueError(f"Parameter {key} must be a scalar or a 1D array, got shape {value.shape}.")

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

		# Check kwargs
		kwargs = self.check_kwargs(**kwargs)

		# Call the appropriate method
		if jnp.isscalar(x1) and jnp.isscalar(x2):
			return self.compute_scalar_if_not_nan(x1, x2, **kwargs)
		elif jnp.ndim(x1) == 1 and jnp.isscalar(x2):
			return self.compute_vector_if_not_nan(x1, x2, **kwargs)
		elif jnp.isscalar(x1) and jnp.ndim(x2) == 1:
			return self.compute_vector_if_not_nan(x2, x1, **kwargs)
		elif jnp.ndim(x1) == 1 and jnp.ndim(x2) == 1:
			return self.compute_matrix(x1, x2, **kwargs)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 2:
			return self.compute_batch(x1, x2, **kwargs)
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
	def compute_scalar_if_not_nan(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Returns NaN if either x1 or x2 is NaN, otherwise calls the compute_scalar method.

		:param x1: scalar array
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: scalar array
		"""
		return lax.cond(jnp.isnan(x1) | jnp.isnan(x2), lambda _: jnp.nan,
		                lambda _: self.compute_scalar(x1, x2, **kwargs), None)

	@jit
	def compute_scalar(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between two scalar arrays.

		:param x1: scalar array
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: scalar array
		"""
		return jnp.array(jnp.nan)  # To be overwritten in subclasses

	@jit
	def compute_vector(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between a vector and a scalar.

		:param x1: vector array (N, )
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: vector array (N, )
		"""
		return vmap(lambda x: self.compute_scalar_if_not_nan(x, x2, **kwargs), in_axes=0)(x1)

	@jit
	def compute_vector_if_not_nan(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Returns an array of NaN if scalar is NaN, otherwise calls the compute_vector method.

		:param x1: vector array (N, )
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: vector array (N, )
		"""
		return lax.cond(jnp.any(jnp.isnan(x2)), lambda _: x1 * jnp.nan, lambda _: self.compute_vector(x1, x2, **kwargs),
		                None)

	@jit
	def compute_matrix(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the kernel covariance matrix between two vector arrays.

		:param x1: vector array (N, )
		:param x2: vector array (M, )
		:param kwargs: hyperparameters of the kernel
		:return: matrix array (N, M)
		"""
		return vmap(lambda x: self.compute_vector_if_not_nan(x2, x, **kwargs), in_axes=0)(x1)

	@jit
	def compute_batch(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the kernel covariance matrix between two batched vector arrays.

		:param x1: vector array (B, N)
		:param x2: vector array (B, M)
		:param kwargs: hyperparameters of the kernel. Each HP that is a scalar will be common to the whole batch, and
		each HP that is a vector will be distinct and thus must have shape (B, )
		:return: tensor array (B, N, M)
		"""
		# vmap(self.compute_matrix)(x1, x2, **kwargs)
		common_hps = {key: value for key, value in kwargs.items() if jnp.isscalar(value)}
		distinct_hps = {key: value for key, value in kwargs.items() if not jnp.isscalar(value)}

		return vmap(lambda x, y, hps: self.compute_matrix(x, y, **hps, **common_hps), in_axes=(0, 0, 0))(x1, x2,
		                                                                                                 distinct_hps)
