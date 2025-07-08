import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class
from jax.lax import cond

from Kernax import AbstractKernel, ConstantKernel


@register_pytree_node_class
class WrapperKernel(AbstractKernel):
	""" Class for kernels that perform some operation on the output of another "inner" kernel."""
	def __init__(self, inner_kernel=None, **kwargs):
		"""
		Instantiates a wrapper kernel with the given inner kernel.

		:param inner_kernel: the inner kernel to wrap
		:param kwargs: hyperparameters of the kernel
		"""
		# If the inner kernel is not a kernel, we try to convert it to a ConstantKernel
		if not isinstance(inner_kernel, AbstractKernel):
			inner_kernel = ConstantKernel(value=inner_kernel)

		self.inner_kernel = inner_kernel

		super().__init__(**kwargs)


@register_pytree_node_class
class DiagKernel(WrapperKernel):
	"""
	Kernel that returns a value only if the inputs are equal, otherwise returns 0.
	This results in a diagonal cross-covariance matrix.
	"""
	@jit
	def pairwise_cov(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		return cond(jnp.all(x1 == x2),
		            lambda _: self.inner_kernel(x1, x2),
		            lambda _: 0.
		            , None)


@register_pytree_node_class
class ExpKernel(WrapperKernel):
	"""
	Kernel that applies the exponential operator to the output of another kernel.
	"""
	@jit
	def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray = None) -> jnp.ndarray:
		if x2 is None:
			x2 = x1

		return jnp.exp(self.inner_kernel(x1, x2))


@register_pytree_node_class
class LogKernel(WrapperKernel):
	"""
	Kernel that applies the logarithm operator to the output of another kernel.
	"""
	@jit
	def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray = None) -> jnp.ndarray:
		if x2 is None:
			x2 = x1

		return jnp.log(self.inner_kernel(x1, x2))


@register_pytree_node_class
class NegKernel(WrapperKernel):
	@jit
	def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray = None) -> jnp.ndarray:
		if x2 is None:
			x2 = x1

		return - self.inner_kernel(x1, x2)
