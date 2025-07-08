import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class

from Kernax import AbstractKernel, ConstantKernel


@register_pytree_node_class
class OperatorKernel(AbstractKernel):
	""" Class for kernels that apply some operation on the output of two kernels."""
	def __init__(self, left_kernel, right_kernel, **kwargs):
		"""
		Instantiates a sum kernel with the given kernels.

		:param right_kernel: the right kernel to sum
		:param left_kernel: the left kernel to sum
		"""
		# If any of the kernels is numerical, we create a ConstantKernel
		if isinstance(left_kernel, (int, float)):
			left_kernel = ConstantKernel(value=left_kernel)
		if isinstance(right_kernel, (int, float)):
			right_kernel = ConstantKernel(value=right_kernel)

		self.left_kernel = left_kernel
		self.right_kernel = right_kernel

		super().__init__(**kwargs)


@register_pytree_node_class
class SumKernel(OperatorKernel):
	""" Sum kernel that sums the outputs of two kernels."""

	@jit
	def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray = None) -> jnp.ndarray:
		if x2 is None:
			x2 = x1

		return self.left_kernel(x1, x2) + self.right_kernel(x1, x2)


@register_pytree_node_class
class ProductKernel(OperatorKernel):
	""" Product kernel that multiplies the outputs of two kernels. """

	@jit
	def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray = None) -> jnp.ndarray:
		if x2 is None:
			x2 = x1

		return self.left_kernel(x1, x2) * self.right_kernel(x1, x2)
