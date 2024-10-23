class TorchBackend(DefaultNumPyBackend):
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