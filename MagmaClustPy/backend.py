import logging
from typing import Tuple

# import jax.numpy as jnp
import mlx.core as mx  # Placeholder import for illustration
import numpy as np
import torch


class DefaultNumPyLinearAlgebraBackend:
	"""
	DefaultNumPyLinearAlgebraBackend provides default implementations for basic
	linear algebra operations using NumPy. This class serves as a base class
	for other backends.

	:ivar array_type: The type of array used by the backend. Default is `numpy.ndarray`.
	:vartype array_type: type

	Methods
	-------
	matmul(a, b)
		Performs matrix multiplication using NumPy.
	inv(a)
		Computes the inverse of a matrix using NumPy.
	pinv(a)
		Computes the pseudo-inverse of a matrix using NumPy.
	"""

	array_type = np.ndarray
	seed = 42
	np.random.seed(seed)

	@staticmethod
	def range(start, end, step):
		"""
		Generates a range of numbers using NumPy.

		:param start: The starting value of the range.
		:type start: int
		:param end: The end value of the range.
		:type end: int
		:param step: The step size between values.
		:type step: float

		:returns: An array of numbers in the specified range.
		:rtype: numpy.ndarray
		"""
		return np.arange(start, end, step)

	@staticmethod
	def sample(array, size, replace=False):
		"""
		Samples elements from an array using NumPy.

		:param array: The array to sample from, must be 1D.
		:type array: array_like
		:param size: The number of samples to draw.
		:type size: int
		:param replace: Whether to sample with replacement. Default is False.
		:type replace: bool, optional

		:returns: The sampled elements.
		:rtype: numpy.ndarray
		"""
		assert len(array.shape) == 1, "Input array must be 1D."
		return np.random.choice(array, size=size, replace=replace)

	@staticmethod
	def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
		"""
		Performs matrix multiplication using NumPy.

		:param a: First matrix to be multiplied.
		:type a: numpy.ndarray
		:param b: Second matrix to be multiplied.
		:type b: numpy.ndarray

		:returns: The result of the matrix multiplication.
		:rtype: numpy.ndarray
		"""
		return np.dot(a, b)

	@staticmethod
	def inv(a: np.ndarray) -> np.ndarray:
		"""
		Computes the inverse of a matrix using NumPy.

		:param a: Matrix to be inverted.
		:type a: numpy.ndarray

		:returns: The inverse of the input matrix.
		:rtype: numpy.ndarray
		"""
		return np.linalg.inv(a)

	@staticmethod
	def pinv(a: np.ndarray) -> np.ndarray:
		"""
		Computes the pseudo-inverse of a matrix using NumPy.

		:param a: Matrix to be inverted.
		:type a: numpy.ndarray

		:returns: The pseudo-inverse of the input matrix.
		:rtype: numpy.ndarray
		"""
		return np.linalg.pinv(a)

	@staticmethod
	def draw(interval: Tuple[float, float]) -> float:
		"""
		Draw uniformly a number within a specified interval.

		:param interval: An interval of values we want to draw uniformly in.
		:type interval: tuple of float

		:returns: A 2-decimals-rounded random number.
		:rtype: float

		:examples:

			Draw a number within the interval (1.0, 2.0)::

				>>> DefaultNumPyLinearAlgebraBackend.draw((1.0, 2.0))
		"""
		# TODO(enhancement): evaluate if rounding is necessary
		return round(np.random.uniform(interval[0], interval[1]), 2)

	@staticmethod
	def zeros(shape) -> np.ndarray:
		"""
		Generates an array of zeros using NumPy.

		:param shape: The shape of the array.
		:type shape: tuple

		:returns: An array of zeros with the specified shape.
		:rtype: numpy.ndarray
		"""
		return np.zeros(shape)

	@staticmethod
	def sum(array, axis=None) -> float:
		"""
		Computes the sum of an array using NumPy.

		:param array: The array to sum.
		:type array: array_like
		:param axis: Axis or axes along which a sum is performed. Default is None.
		:type axis: int or tuple of int, optional

		:returns: The sum of the input array.
		:rtype: float
		"""
		return np.sum(array, axis=axis)

	@staticmethod
	def exp(array) -> np.ndarray:
		"""
		Computes the element-wise exponential of an array using NumPy.

		:param array: The array to exponentiate.
		:type array: array_like

		:returns: The element-wise exponential of the input array.
		:rtype: numpy.ndarray
		"""
		return np.exp(array)

	@staticmethod
	def dot(a, b) -> float:
		"""
		Computes the dot product of two arrays using NumPy.

		:param a: First array.
		:type a: array_like
		:param b: Second array.
		:type b: array_like

		:returns: The dot product of the two input arrays.
		:rtype: float
		"""
		return np.dot(a, b)

	@staticmethod
	def sort(array) -> np.ndarray:
		"""
		Sorts an array using NumPy.

		:param array: The array to sort.
		:type array: array_like

		:returns: The sorted array.
		:rtype: numpy.ndarray
		"""
		return np.sort(array)


# class JaxBackend(DefaultNumPyLinearAlgebraBackend):
#     """
#     JaxBackend provides implementations for basic linear algebra operations using JAX.
#
#     Methods
#     -------
#     matmul(a, b)
#         Performs matrix multiplication using JAX.
#     inv(a)
#         Computes the inverse of a matrix using JAX.
#     """
#
#     def matmul(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
#         """
#         Performs matrix multiplication using JAX.
#
#         Parameters
#         ----------
#         a : array_like
#             First matrix to be multiplied.
#         b : array_like
#             Second matrix to be multiplied.
#
#         Returns
#         -------
#         jax.numpy.ndarray
#             The result of the matrix multiplication.
#         """
#         return jnp.dot(a, b)
#
#     def inv(self, a: jnp.ndarray) -> jnp.ndarray:
#         """
#         Computes the inverse of a matrix using JAX.
#
#         Parameters
#         ----------
#         a : array_like
#             Matrix to be inverted.
#
#         Returns
#         -------
#         jax.numpy.ndarray
#             The inverse of the input matrix.
#         """
#         return jnp.linalg.inv(a)
#
#     def pinv(self, a: jnp.ndarray) -> jnp.ndarray:
#         """
#         Computes the pseudo-inverse of a matrix using JAX.
#
#         Parameters
#         ----------
#         a : array_like
#             Matrix to be inverted.
#
#         Returns
#         -------
#         jax.numpy.ndarray
#             The pseudo-inverse of the input matrix.
#         """
#         logging.warning("Pseudo-inverse not yet implemented for backend 'JAX'. Falling back to NumPy implementation.")
#         # Convert to numpy array
#         a = a.asnumpy()
#         # Call superclass implementation
#         res = super().pinv(a)
#         # Convert back to JAX
#         return jnp.array(res)


class MLXBackend(DefaultNumPyLinearAlgebraBackend):
	"""
	MLXBackend provides implementations for basic linear algebra operations using MLX.

	Methods
	-------
	matmul(a, b)
		Performs matrix multiplication using MLX.
	inv(a)
		Computes the inverse of a matrix using MLX.
	"""

	def matmul(self, a: mx.array, b: mx.array) -> mx.array:
		"""
		Performs matrix multiplication using MLX.

		Parameters
		----------
		a : array_like
			First matrix to be multiplied.
		b : array_like
			Second matrix to be multiplied.

		Returns
		-------
		mlx.ndarray
			The result of the matrix multiplication.
		"""
		return a @ b

	def inv(self, a: mx.array) -> mx.array:
		"""
		Computes the inverse of a matrix using MLX.

		Parameters
		----------
		a : array_like
			Matrix to be inverted.

		Returns
		-------
		mlx.ndarray
			The inverse of the input matrix.
		"""
		return mx.linalg.inv(a)


class TorchBackend(DefaultNumPyLinearAlgebraBackend):
	"""
	TorchBackend provides implementations for basic linear algebra operations using PyTorch.

	Methods
	-------
	matmul(a, b)
		Performs matrix multiplication using PyTorch.
	inv(a)
		Computes the inverse of a matrix using PyTorch.
	"""

	def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		"""
		Performs matrix multiplication using PyTorch.

		Parameters
		----------
		a : array_like
			First matrix to be multiplied.
		b : array_like
			Second matrix to be multiplied.

		Returns
		-------
		torch.Tensor
			The result of the matrix multiplication.
		"""
		return torch.matmul(a, b)

	def inv(self, a: torch.Tensor) -> torch.Tensor:
		"""
		Computes the inverse of a matrix using PyTorch.

		Parameters
		----------
		a : array_like
			Matrix to be inverted.

		Returns
		-------
		torch.Tensor
			The inverse of the input matrix.
		"""
		return torch.inverse(a)
