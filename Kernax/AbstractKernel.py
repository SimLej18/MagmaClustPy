from jax import jit, vmap
from jax.tree_util import register_pytree_node_class
from jax.lax import cond
import jax.numpy as jnp


@register_pytree_node_class
class AbstractKernel:
	def __init__(self, skip_check=False, **kwargs):
		if not skip_check:
			# Check that hyperparameters are all jnp arrays/scalars or kernels
			for key, value in kwargs.items():
				if not isinstance(value, jnp.ndarray):  # Check type
					kwargs[key] = jnp.array(value)
				if len(kwargs[key].shape) > 1:  # Check dimensionality
					# TODO: this could be more flexible, e.g. allow different lengthscales for different dimensions
					#  but then we cannot guess if HPs are shared only based on the shape
					#  This could be fixed by always using 3D arrays for hyperparameters, and use broadcasting at
					#  every step of the process
					raise ValueError(f"Parameter {key} must be a scalar or a 1D array, got shape {value.shape}.")

		# Register hyperparameters in *kwargs* as instance attributes
		self.__dict__.update(kwargs)

	def __str__(self):
		return f"{self.__class__.__name__}({', '.join([f'{key}={value}' for key, value in self.__dict__.items()])})"

	def __repr__(self):
		return str(self)

	@jit
	def check_kwargs(self, **kwargs):
		"""
		This method is called everytime the kernel is called, to ensure that all hyperparameters are present.
		It loads the hyperparameters from the instance attributes if they are not provided in kwargs.

		That way, attributes from the kernel instance can be seen as default hyperparameters, and the user can
		override them by passing them in the kwargs.

		:param kwargs: Hyperparameters of the kernel, as a dictionary. As this method is defined in the superclass,
		we can't know in advance the names and values of the hyperparameters that will be used.
		:return: Completed kwargs dictionary with all hyperparameters.
		"""
		for key in self.__dict__:
			if key not in kwargs:
				kwargs[key] = self.__dict__[key]
		return kwargs

	@jit
	def __call__(self, x1, x2=None, **kwargs):
		"""
		Computes the (cross) covariance between two inputs x1 and x2 using the kernel.
		This method automatically adapts to the dimensionality of the inputs and calls the appropriate sub-method.

		:param x1: The first input, can be a scalar, vector or matrix.
		:param x2: The second input, can be a scalar, vector or matrix. If None, the covariance between x1 and itself
		is computed.
		:param kwargs: Hyperparameters of the kernel, as a dictionary. For each hyperparameter that is not provided, the
		value stored in the kernel instance will be used.

		:return: The covariance value(s) between x1 and x2, as a scalar, vector or matrix, depending on the inputs.
		"""
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
		return cls(*children, skip_check=True)

	@jit
	def pairwise_cov(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the kernel pairwise covariance value between two input vectors.

		:param x1: Input vector, shape (I,)
		:param x2: Input vector, shape (I,)
		:param kwargs: Hyperparameters of the kernel
		:return: Scalar covariance value
		"""
		return jnp.array(jnp.nan)  # To be overwritten in subclasses

	@jit
	def pairwise_cov_if_not_nan(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Returns NaN if either x1 or x2 contains NaNs, otherwise calls the pairwise_cov method.

		:param x1: Input vector, shape (I,)
		:param x2: Input vector, shape (I,)
		:param kwargs: Hyperparameters of the kernel
		:return: Scalar covariance value or NaN
		"""
		return cond(jnp.any(jnp.isnan(x1) | jnp.isnan(x2)), lambda _: jnp.nan,
		            lambda _: self.pairwise_cov(x1, x2, **kwargs), None)

	@jit
	def cross_cov_vector(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the kernel cross-covariance values between an array of vectors (matrix) and a vector.

		:param x1: Array of input vectors, shape (N, I)
		:param x2: input vector, shape (I,)
		:param kwargs: Hyperparameters of the kernel
		:return: Values of the cross-covariance between each vector in x1 and the vector x2, shape (N,)
		"""
		return vmap(lambda x: self.pairwise_cov_if_not_nan(x, x2, **kwargs), in_axes=0)(x1)

	@jit
	def cross_cov_vector_if_not_nan(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Returns an array of NaN if x2 contains NaNs, otherwise calls the compute_vector method.

		:param x1: Array of input vectors, shape (N, I)
		:param x2: Input vector, shape (I,)
		:param kwargs: Hyperparameters of the kernel
		:return: Values of the cross-covariance between each vector in x1 and the vector x2, or NaNs, shape (N,)
		"""
		return cond(jnp.any(jnp.isnan(x2)), lambda _: jnp.full(len(x1), jnp.nan), lambda _: self.cross_cov_vector(x1, x2, **kwargs),
		            None)

	@jit
	def cross_cov_matrix(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the cross-covariance matrix between two vector arrays.

		:param x1: Array of input vectors, shape (N, I)
		:param x2: Array of input vectors, shape (M, I)
		:param kwargs: Hyperparameters of the kernel
		:return: Cross-covariance matrix, shape (N, M)
		"""
		return vmap(lambda x: self.cross_cov_vector_if_not_nan(x2, x, **kwargs), in_axes=0)(x1)

	@jit
	def cross_cov_batch(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the kernel covariance matrix between two batched vector arrays.

		:param x1: Batch of array of input vectors, shape (T, N, I)
		:param x2: Batch of array of input vectors, shape (T, M, I)
		:param kwargs: hyperparameters of the kernel. Each HP that is a scalar will be shared to the whole batch, and
		each HP that is a vector will be distinct and thus must have shape (T,)

		:return: Batch of all cross-covariances (B, N, M)
		"""
		# vmap(self.compute_matrix)(x1, x2, **kwargs)
		shared_hps = {key: value for key, value in kwargs.items() if jnp.isscalar(value)}
		distinct_hps = {key: value for key, value in kwargs.items() if not jnp.isscalar(value)}

		return vmap(lambda x, y, hps: self.cross_cov_matrix(x, y, **hps, **shared_hps), in_axes=(0, 0, 0))(x1, x2, distinct_hps)
