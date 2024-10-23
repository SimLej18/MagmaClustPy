import logging

import mlx.core as mx
import numpy as np

from MagmaClustPy.backend import DefaultNumPyBackend


class MLXArray:
	"""
	Class for mirroring and extending mx.ndarray
	"""

	def __getattr__(self, item):
		"""
		Forward method calls to np.ndarray.

		:param item: The name of the method to call.
		:type item: str

		:returns: The result of the method call.
		"""
		return getattr(mx.array, item)


class MLXRandom:
	"""
	Class for mirroring and extending mx.random
	"""

	def __getattr__(self, item):
		"""
		Forward method calls to np.random.

		:param item: The name of the method to call.
		:type item: str

		:returns: The result of the method call.
		"""
		return getattr(mx.random, item)


class MLXLinearAlgebra:
	"""
	Class for mirroring and extending mx.linalg
	"""

	def __getattr__(self, item):
		"""
		Forward method calls to np.linalg.

		:param item: The name of the method to call.
		:type item: str

		:returns: The result of the method call.
		"""
		return getattr(mx.linalg, item)


class MLXBackend(DefaultNumPyBackend):
	"""
	MLXBackend provides implementations for basic linear algebra operations using MLX.
	The goal of this back-end is to have the same interface as the NumPy back-end, so they can
	be used interchangeably in the MagmaClustPy package.

	Methods
	-------
	matmul(a, b)
		Performs matrix multiplication using MLX.
	"""
	array = MLXArray()
	random = MLXRandom()
	linalg = MLXLinearAlgebra()

	def __getattr__(self, item):
		"""
		Forward method calls to mlx or revert to the numpy implementation.

		Parameters
		----------
		item : str
			The name of the method to call.

		Returns
		-------
		Any
			The result of the method call.
		"""
		try:
			return getattr(mx, item)
		except AttributeError:
			logging.warning(f"Method {item} not found in MLX. Reverting to NumPy implementation.")
			try:
				return MLXBackend.wrapper(function=super().__getattr__(item))
			except AttributeError:
				raise AttributeError(f"Method {item} not found in MLX or NumPy.")

	@staticmethod
	def wrapper(*args, function=None, **kwargs):
		"""
		Wrapper to be used when an implementation is not available in the MLX backend.
		Converts MLX arrays to NumPy, calls the function, and converts back to MLX.
		"""
		# Convert MLX arrays to NumPy arrays
		converted_args = [np.asarray(arg) if isinstance(arg, mx.array) else arg for arg in args]
		converted_kwargs = {k: np.asarray(v) if isinstance(v, mx.array) else v for k, v in kwargs.items()}

		# Call the original NumPy function
		result = function(*converted_args, **converted_kwargs)

		# Convert the result back to MLX array if needed
		if isinstance(result, np.ndarray):
			return mx.array(result)
		return result

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