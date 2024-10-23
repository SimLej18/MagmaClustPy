import logging

import numpy as np

from MagmaClustPy import lin_alg_backend as lab
from MagmaClustPy import config


class Kernel:
	"""
	Kernel interface for defining custom kernels.

	Attributes:
		params (dict): Dictionary of hyperparameters for the kernel.
	"""
	hp_min = 0
	hp_max = 3
	noise_min = -5
	noise_max = -1

	def __init__(self, **params):
		"""
		Initialize the kernel with hyperparameters.

		:param params: Hyperparameters for the kernel.
		"""
		self.params = params.keys()

	def __call__(self, x1: float, x2: float) -> float:
		"""
		Compute the kernel value between two inputs x1 and x2.

		:param x1: First input (can be a scalar or a vector).
		:param x2: Second input (can be a scalar or a vector).
		:return: The kernel value between x1 and x2.
		"""
		raise NotImplementedError

	def compute_matrix(self, x1: lab.array.type, x2: lab.array.type = None) -> lab.array.type:
		"""
		Compute the kernel matrix between two sets of inputs.

		:param x1: First set of inputs (n_samples_1, n_features).
		:param x2: Second set of inputs (n_samples_2, n_features). If None, X2 = X1.
		:return: The kernel matrix (n_samples_1, n_samples_2).
		"""
		x2 = x2 if x2 is not None else x1
		K = np.zeros((x1.shape[0], x2.shape[0]))
		for i in range(x1.shape[0]):
			for j in range(x2.shape[0]):
				K[i, j] = self.__call__(x1[i], x2[j])
		return K

	def inverse_covariance_matrix(self, x, pen_diag=config["pen_diag"]):
		"""
		Compute the inverse covariance matrix of the kernel for a set of inputs.
		Equivalent to the `kern_to_inv()` function from MagmaClustR.

		:param x: Set of inputs (n_samples,). A 1D array.
		:param pen_diag: A jitter term, added on the diagonal to prevent numerical issues when inverting nearly singular matrices.
		:return: The inverse covariance matrix (n_samples, n_samples).
		"""
		k = self.compute_matrix(x)
		return lab.linalg.inv(k + pen_diag * lab.eye(k.shape[0]))


class SquaredExponentialKernel(Kernel):
	"""
	Squared Exponential Kernel (Gaussian Kernel).

	This kernel is used in various machine learning algorithms, particularly in Gaussian Processes.
	It computes the similarity between two points based on their distance in the input space.

	Attributes:
		length_scale (float): Determines how quickly the function decays as points move apart.
		variance (float): Determines the amplitude of the kernel.
	"""

	def __init__(self, length_scale=None, variance=None):
		"""
		Initialize the kernel with length_scale and variance parameters.

		:param length_scale: Determines how quickly the function decays as points move apart (default: 1.0).
		:param variance: Determines the amplitude of the kernel (default: 1.0).
		"""
		super().__init__(length_scale=length_scale, variance=variance)

		if length_scale is None:
			length_scale = lab.random.uniform(self.hp_min, self.hp_max)
		if variance is None:
			variance = lab.random.uniform(self.hp_min, self.hp_max)

		self.length_scale = length_scale
		self.variance = variance

		logging.info(f"SE kernel initialized with length_scale={length_scale} and variance={variance}")

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


class SquaredExponentialMagmaKernel(Kernel):
	"""
	As of its version 1.2, MagmaClustR doesn't use the standard SE Kernel. It uses a variation given by this formula:
	k(x, x') = exp(var - exp(-len) * (x - x')^2 * 0.5)
	"""
	def __init__(self, length_scale=None, variance=None):
		"""
		Initialize the kernel with length_scale and variance parameters.

		:param length_scale: Determines how quickly the function decays as points move apart (default: 1.0).
		:param variance: Determines the amplitude of the kernel (default: 1.0).
		"""
		super().__init__(length_scale=length_scale, variance=variance)

		if length_scale is None:
			length_scale = lab.random.uniform(self.hp_min, self.hp_max)
		if variance is None:
			variance = lab.random.uniform(self.hp_min, self.hp_max)

		self.length_scale = length_scale
		self.variance = variance

		logging.info(f"SE kernel initialized with length_scale={length_scale} and variance={variance}")

	def __call__(self, x1, x2):
		"""
		Compute the squared exponential kernel between two inputs x1 and x2.

		:param x1: First input (can be a scalar or a vector).
		:param x2: Second input (can be a scalar or a vector).
		:return: The kernel value between x1 and x2.
		"""
		dist_sq = np.sum((x1 - x2) ** 2)
		return np.exp(self.variance - np.exp(-self.length_scale) * dist_sq * 0.5)
