from typing import Tuple

import numpy as np
from scipy.optimize import minimize


class NumpyArray:
	"""
	Class for mirroring and extending np.ndarray.
	"""
	type = np.ndarray

	def __getattr__(self, item):
		"""
		Forward method calls to np.ndarray.

		:param item: The name of the method to call.
		:type item: str

		:returns: The result of the method call.
		"""
		return getattr(np.ndarray, item)

	def __call__(self, *args, **kwargs):
		# TODO: evaluate if we should use np.array or np.ndarray by default
		return np.array(*args, **kwargs)


class NumpyRandom:
	"""
	Class for generating random numbers using NumPy, to be used inside the NumPy backend.
	If the function is not defined inside the class, it is forwarded to the NumPy random module.
	"""

	@staticmethod
	def sample(array: NumpyArray, size, replace=False):
		"""
		Samples elements from an array using NumPy.

		:param array: The array to sample from, must be 1D.
		:type array: NumpyArray (but should work for any array-like object)
		:param size: The number of samples to draw.
		:type size: int
		:param replace: Whether to sample with replacement. Default is False.
		:type replace: bool, optional

		:returns: The sampled elements.
		:rtype: NumpyArray
		"""
		assert len(array.shape) == 1, "Input array must be 1D."
		return np.random.choice(array, size=size, replace=replace)

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

				>>> DefaultNumPyBackend.random.draw((1.0, 2.0))
		"""
		# TODO(enhancement): rounding is done to imitate Magma but is it really necessary?
		return round(np.random.uniform(interval[0], interval[1]), 2)

	def __getattr__(self, name):
		"""
		Forward method calls to np.random.

		:param name: The name of the method to call.
		:type name: str

		:returns: The result of the method call.
		"""
		return getattr(np.random, name)


class NumpyOptimiser:
	def minimise(self, *args, **kwargs):
		minimize(*args, **kwargs)


class NumpyLinearAlgebra:
	"""
	Class for mirroring and extending np.linalg
	"""

	def __getattr__(self, item):
		"""
		Forward method calls to np.linalg.

		:param item: The name of the method to call.
		:type item: str

		:returns: The result of the method call.
		"""
		return getattr(np.linalg, item)


class DefaultNumPyBackend:
	"""
	DefaultNumPyBackend provides default implementations for basic
	linear algebra operations using NumPy. This class serves as a base class
	for other backends.

	Methods
	-------
	matmul(a, b)
		Performs matrix multiplication using NumPy.
	inv(a)
		Computes the inverse of a matrix using NumPy.
	pinv(a)
		Computes the pseudo-inverse of a matrix using NumPy.
	"""

	array = NumpyArray()
	random = NumpyRandom()
	random.seed(42)
	linalg = NumpyLinearAlgebra()
	optim = NumpyOptimiser()

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

	def __getattr__(self, item):
		"""
		Forward method calls to np.

		:param item: The name of the method to call.
		:type item: str

		:returns: The result of the method call.
		"""
		return getattr(np, item)
