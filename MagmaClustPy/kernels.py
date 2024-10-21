import numpy as np


class SquaredExponentialKernel:
	"""
	Squared Exponential Kernel (Gaussian Kernel).

	This kernel is used in various machine learning algorithms, particularly in Gaussian Processes.
	It computes the similarity between two points based on their distance in the input space.

	Attributes:
		length_scale (float): Determines how quickly the function decays as points move apart.
		variance (float): Determines the amplitude of the kernel.
	"""

	def __init__(self, length_scale=1.0, variance=1.0):
		"""
		Initialize the kernel with length_scale and variance parameters.

		:param length_scale: Determines how quickly the function decays as points move apart (default: 1.0).
		:param variance: Determines the amplitude of the kernel (default: 1.0).
		"""
		self.length_scale = length_scale
		self.variance = variance

	def __call__(self, x1, x2):
		"""
		Compute the squared exponential kernel between two inputs x1 and x2.

		:param x1: First input (can be a scalar or a vector).
		:param x2: Second input (can be a scalar or a vector).
		:return: The kernel value between x1 and x2.
		"""
		dist_sq = np.sum((x1 - x2) ** 2)
		return self.variance * np.exp(-0.5 * dist_sq / self.length_scale ** 2)

	def compute_matrix(self, x1, x2=None):
		"""
		Compute the kernel matrix between two sets of inputs using vectorized operations.

		:param x1: First set of inputs (n_samples_1,). A 1D array.
		:param x2: Second set of inputs (n_samples_2,). A 1D array. If None, X2 = X1.
		:return: The kernel matrix (n_samples_1, n_samples_2).
		"""
		# If X2 is None, use X1 for both inputs
		x2 = x2 if x2 is not None else x1

		# Reshape X1 and X2 to 2D arrays (n_samples, 1) to broadcast correctly
		x1 = x1[:, np.newaxis]  # Shape becomes (n_samples_1, 1)
		x2 = x2[:, np.newaxis]  # Shape becomes (n_samples_2, 1)

		# Compute pairwise squared Euclidean distances between points in X1 and X2
		dist_sq = np.sum(x1 ** 2, axis=1)[:, np.newaxis] + np.sum(x2 ** 2, axis=1) - 2 * np.dot(x1, x2.T)

		# Apply the squared exponential kernel function to the distance matrix
		k = self.variance * np.exp(-0.5 * dist_sq / self.length_scale ** 2)

		return k
