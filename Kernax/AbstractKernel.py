import jax.numpy as jnp
from jax import jit, vmap
from jax.tree_util import register_pytree_node_class, tree_map
from jax.tree import reduce
from jax.lax import cond

from functools import partial


class StaticAbstractKernel:
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
		return jnp.array(jnp.nan)  # To be overwritten in subclasses

	@classmethod
	@partial(jit, static_argnums=(0,))
	def pairwise_cov_if_not_nan(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Returns NaN if either x1 or x2 is NaN, otherwise calls the compute_scalar method.

		:param kern: the kernel to use, containing hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
		return cond(jnp.any(jnp.isnan(x1) | jnp.isnan(x2)),
		            lambda _: jnp.nan,
		            lambda _: cls.pairwise_cov(kern, x1, x2),
		            None)

	@classmethod
	@partial(jit, static_argnums=(0,))
	def cross_cov_vector(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel cross covariance values between an array of vectors (matrix) and a vector.

		:param kern: the kernel to use, containing hyperparameters
		:param x1: vector array (N, )
		:param x2: scalar array
		:return: vector array (N, )
		"""
		return vmap(lambda x: cls.pairwise_cov_if_not_nan(kern, x, x2), in_axes=0)(x1)

	@classmethod
	@partial(jit, static_argnums=(0,))
	def cross_cov_vector_if_not_nan(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Returns an array of NaN if scalar is NaN, otherwise calls the compute_vector method.

		:param kern: the kernel to use, containing hyperparameters
		:param x1: vector array (N, )
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: vector array (N, )
		"""
		return cond(jnp.any(jnp.isnan(x2)),
		            lambda _: jnp.full(len(x1), jnp.nan),
		            lambda _: cls.cross_cov_vector(kern, x1, x2),
		            None)

	@classmethod
	@partial(jit, static_argnums=(0,))
	def cross_cov_matrix(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel covariance matrix between two vector arrays.

		:param x1: vector array (N, )
		:param x2: vector array (M, )
		:return: matrix array (N, M)
		"""
		return vmap(lambda x: cls.cross_cov_vector_if_not_nan(kern, x2, x), in_axes=0)(x1)

	@classmethod
	@partial(jit, static_argnums=(0,))
	def cross_cov_batch(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
		"""
		Compute the kernel covariance matrix between two batched vector arrays.

		:param x1: vector array (B, N)
		:param x2: vector array (B, M)
		:return: tensor array (B, N, M)
		"""
		has_input_dim = lambda param: hasattr(param, 'shape') and len(param.shape) > 0 and param.shape[0] == x1.shape[0]
		hp_vmap = tree_map(lambda param: 0 if has_input_dim(param) else None, kern)

		return vmap(lambda k, x, y: cls.cross_cov_matrix(k, x, y), in_axes=(hp_vmap, 0, 0))(kern, x1, x2)


@register_pytree_node_class
class AbstractKernel:
	def __init__(self, **kwargs):
		"""
		Instatiates a kernel with the given hyperparameters.
		https://docs.jax.dev/en/latest/pytrees.html#custom-pytrees-and-initialization
		:param kwargs: the hyperparameters of the kernel, as keyword arguments.
		"""

		# Register hyperparameters in *kwargs* as instance attributes
		self.__dict__.update(kwargs)

		# This check allows the user to assign a static class before and after calling super().__init__()
		if not hasattr(self, 'static_class'):
			self.static_class = StaticAbstractKernel

		if not hasattr(self, 'static_attributes'):
			self.static_attributes = {"static_class", "static_attributes"}
		else:
			self.static_attributes.union({"static_class", "static_attributes"})

	def __str__(self):
		return f"{self.__class__.__name__}({', '.join([f'{key}={value}' for key, value in self.__dict__.items() if key not in self.static_attributes])})"

	def __repr__(self):
		return str(self)

	@jit
	def __call__(self, x1, x2=None):
		# If no x2 is provided, we compute the covariance between x1 and itself
		if x2 is None:
			x2 = x1

		# Turn scalar inputs into vectors
		x1, x2 = jnp.atleast_1d(x1), jnp.atleast_1d(x2)

		# Check for distinct hyperparameters
		if self.has_distinct_hyperparameters(x1.shape[0]) and (x1.ndim != 3 or x2.ndim != 3):
			raise ValueError("Kernel with distinct hyperparameters was called on unbatched inputs. It cannot know which hyperparameter value to use for this element")

		# Call the appropriate method
		if jnp.ndim(x1) == 1 and jnp.ndim(x2) == 1:
			return self.static_class.pairwise_cov_if_not_nan(self, x1, x2)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 1:
			return self.static_class.cross_cov_vector_if_not_nan(self, x1, x2)
		elif jnp.ndim(x1) == 1 and jnp.ndim(x2) == 2:
			return self.static_class.cross_cov_vector_if_not_nan(self, x2, x1)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 2:
			return self.static_class.cross_cov_matrix(self, x1, x2)
		elif jnp.ndim(x1) == 3 and jnp.ndim(x2) == 3:
			if x1.shape[0] != x2.shape[0]:
				raise ValueError(f"Batch dimension mismatch: x1 has shape {x1.shape}, x2 has shape {x2.shape}.")
			return self.static_class.cross_cov_batch(self, x1, x2)
		else:
			raise ValueError(
				f"Invalid input dimensions: x1 has shape {x1.shape}, x2 has shape {x2.shape}. "
				"Expected 1D, 2D arrays or 3D arrays for batched inputs."
			)

	def has_distinct_hyperparameters(self, inputs_first_dim) -> bool:
		"""
		Checks if the kernel has distinct hyperparameters based on the first dimension of the inputs.

		:param inputs_first_dim: The first dimension of the inputs to check against the hyperparameters.
		:return: True if the kernel has distinct hyperparameters, False otherwise.
		"""
		return reduce(
			lambda acc, param: acc or (hasattr(param, 'shape') and len(param.shape) > 0 and param.shape[0] == inputs_first_dim)
			, self, False)

	# Methods to use Kernel as a PyTree
	def tree_flatten(self):
		return tuple(val for key, val in self.__dict__.items() if key not in self.static_attributes), None  # No static values

	@classmethod
	def tree_unflatten(cls, _, children):
		# This class being abstract, this function fails when called on an "abstract instance",
		# as we don't know the number of parameters the constructor expects, yet we send it children.
		# On a subclass, this will work as expected as long as the constructor has a clear number of
		# kwargs as parameters.
		return cls(*children)

	def __add__(self, other):
		from Kernax.OperatorKernels import SumKernel
		return SumKernel(self, other)

	def __radd__(self, other):
		from Kernax.OperatorKernels import SumKernel
		return SumKernel(other, self)

	def __neg__(self):
		from Kernax.WrapperKernels import NegKernel
		return NegKernel(self)

	def __mul__(self, other):
		from Kernax.OperatorKernels import ProductKernel
		return ProductKernel(self, other)

	def __rmul__(self, other):
		from Kernax.OperatorKernels import ProductKernel
		return ProductKernel(other, self)
