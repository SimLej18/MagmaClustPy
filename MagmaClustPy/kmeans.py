# COPIED FROM https://github.com/ethqnol/jax-kmeans
# TODO: evaluate if we cannot just import jax-kmeans. At the time of writing this, it's not available through pip.
# See https://github.com/ethqnol/jax-kmeans/issues/1

import warnings
from typing import Optional, Union, Callable, Tuple, Any
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, random, vmap

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, validate_data


# Core JAX utility functions
@jit
def _to_jax_safe(x: jnp.ndarray) -> jnp.ndarray:
	"""Convert to JAX array safely."""
	return jnp.asarray(x)


def _to_numpy_safe(x: jnp.ndarray) -> np.ndarray:
	"""Convert JAX array to numpy safely."""
	return np.asarray(x)


@jit
def _squared_distances_vectorized(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
	"""Vectorized squared euclidean distances with memory optimization."""
	# More memory efficient: avoid creating large intermediate arrays
	return jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)


@jit
def _assign_labels_fast(X: jnp.ndarray, centers: jnp.ndarray) -> jnp.ndarray:
	"""Fast label assignment using vectorized operations."""
	return jnp.argmin(_squared_distances_vectorized(X, centers), axis=1)


@jit
def _compute_inertia_fast(X: jnp.ndarray, centers: jnp.ndarray, labels: jnp.ndarray) -> jnp.float32:
	"""Fast inertia computation with vectorized operations."""
	assigned_centers = centers[labels]
	return jnp.sum(jnp.sum((X - assigned_centers) ** 2, axis=1))


@jit
def _weighted_inertia_fast(X: jnp.ndarray, centers: jnp.ndarray, labels: jnp.ndarray,
                           weights: jnp.ndarray) -> jnp.float32:
	"""Fast weighted inertia computation."""
	assigned_centers = centers[labels]
	distances_sq = jnp.sum((X - assigned_centers) ** 2, axis=1)
	return jnp.sum(distances_sq * weights)


# Use vmap for parallel operations across clusters
@partial(jit, static_argnames=['n_clusters'])
def _update_centers_vectorized(X: jnp.ndarray, labels: jnp.ndarray, weights: jnp.ndarray,
                               n_clusters: int) -> jnp.ndarray:
	"""Vectorized center updates using JAX vmap for maximum performance."""

	def update_single_center(k):
		mask = labels == k
		total_weight = jnp.sum(weights * mask)
		# Handle empty clusters
		return jnp.where(
			total_weight > 0,
			jnp.sum(X * (weights * mask)[:, None], axis=0) / total_weight,
			jnp.zeros(X.shape[1])
		)

	# Vectorize across all clusters
	return vmap(update_single_center)(jnp.arange(n_clusters))


@partial(jit, static_argnames=['n_clusters'])
def _lloyd_step_optimized(X: jnp.ndarray, centers: jnp.ndarray, weights: jnp.ndarray, n_clusters: int) -> Tuple[
	jnp.ndarray, jnp.ndarray, jnp.float32]:
	"""Optimized Lloyd step with full vectorization."""
	# Assign labels
	labels = _assign_labels_fast(X, centers)

	# Update centers
	new_centers = _update_centers_vectorized(X, labels, weights, n_clusters)

	# Compute inertia
	inertia = _weighted_inertia_fast(X, new_centers, labels, weights)

	return labels, new_centers, inertia


# Optimized K-means++ initialization

@partial(jit, static_argnames=['n_clusters'])
def _kmeans_plusplus_init_optimized(X: jnp.ndarray, weights: jnp.ndarray, key: jax.random.PRNGKey,
                                    n_clusters: int) -> jnp.ndarray:
	"""Highly optimized k-means++ initialization using JAX random operations."""
	n_samples, n_features = X.shape
	centers = jnp.zeros((n_clusters, n_features))

	# Choose first center randomly
	key, subkey = random.split(key)
	probs = weights / jnp.sum(weights)
	first_idx = random.choice(subkey, n_samples, p=probs)
	centers = centers.at[0].set(X[first_idx])

	# Choose remaining centers
	for i in range(1, n_clusters):
		key, subkey = random.split(key)

		# Compute distances to closest existing center
		distances_sq = jnp.min(_squared_distances_vectorized(X, centers[:i]), axis=1)

		# Probability proportional to squared distance
		probs = distances_sq * weights
		probs = probs / jnp.sum(probs)

		# Choose next center
		next_idx = random.choice(subkey, n_samples, p=probs)
		centers = centers.at[i].set(X[next_idx])

	return centers


# Main optimized KMeans class

class KMeans(BaseEstimator, ClusterMixin, TransformerMixin):
	"""Highly optimized K-means clustering with JAX JIT compilation.

	This implementation provides significant performance improvements over
	standard implementations through JAX JIT compilation, vectorized operations,
	and GPU acceleration.

	Parameters
	----------
	n_clusters : int, default=8
		The number of clusters to form.
	init : {'k-means++', 'random'} or ndarray, default='k-means++'
		Initialization method.
	n_init : int, default=10
		Number of time the k-means algorithm will be run with different
		centroid seeds.
	max_iter : int, default=300
		Maximum number of iterations.
	tol : float, default=1e-4
		Relative tolerance for convergence.
	random_state : int, RandomState instance or None, default=None
		Random state for reproducibility.
	verbose : bool, default=False
		Verbosity mode.
	"""

	def __init__(self,
	             n_clusters: int = 8,
	             init: Union[str, np.ndarray] = 'k-means++',
	             n_init: int = 10,
	             max_iter: int = 300,
	             tol: float = 1e-4,
	             random_state: Optional[int] = None,
	             verbose: bool = False):

		self.n_clusters = n_clusters
		self.init = init
		self.n_init = n_init
		self.max_iter = max_iter
		self.tol = tol
		self.random_state = random_state
		self.verbose = verbose

	def _validate_params(self, X: np.ndarray) -> None:
		"""Validate parameters against input data."""
		if X.shape[0] < self.n_clusters:
			raise ValueError(f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}")

		if self.n_clusters <= 0:
			raise ValueError(f"n_clusters={self.n_clusters} should be > 0")

	def _init_centers(self, X: jnp.ndarray, weights: jnp.ndarray, random_state: np.random.RandomState) -> jnp.ndarray:
		"""Initialize cluster centers."""
		if isinstance(self.init, str):
			if self.init == 'k-means++':
				key = random.PRNGKey(random_state.randint(0, 2 ** 31))
				return _kmeans_plusplus_init_optimized(X, weights, key, self.n_clusters)
			elif self.init == 'random':
				key = random.PRNGKey(random_state.randint(0, 2 ** 31))
				indices = random.choice(key, X.shape[0], (self.n_clusters,),
				                        p=weights / jnp.sum(weights), replace=False)
				return X[indices]
			else:
				raise ValueError(f"Invalid init parameter: {self.init}")
		else:
			# Custom initialization
			centers = jnp.asarray(self.init)
			if centers.shape != (self.n_clusters, X.shape[1]):
				raise ValueError(f"Init centers shape {centers.shape} doesn't match expected "
				                 f"({self.n_clusters}, {X.shape[1]})")
			return centers

	def _fit_single(self, X: jnp.ndarray, weights: jnp.ndarray, random_state: np.random.RandomState) -> Tuple[
		jnp.ndarray, jnp.ndarray, float, int]:
		"""Single k-means run with optimized JAX operations."""
		# Initialize centers
		centers = self._init_centers(X, weights, random_state)

		best_inertia = jnp.inf
		best_labels = None
		best_centers = None

		for iteration in range(self.max_iter):
			# Perform Lloyd step
			labels, new_centers, inertia = _lloyd_step_optimized(X, centers, weights, self.n_clusters)

			if self.verbose:
				print(f"Iteration {iteration}: inertia = {float(inertia):.6f}")

			# Check for convergence
			center_shift = jnp.sum((new_centers - centers) ** 2)
			if center_shift <= self.tol:
				if self.verbose:
					print(f"Converged at iteration {iteration}")
				break

			centers = new_centers

			# Track best solution
			if inertia < best_inertia:
				best_inertia = inertia
				best_labels = labels
				best_centers = centers

		return best_labels, best_centers, float(best_inertia), iteration + 1

	def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
	        sample_weight: Optional[np.ndarray] = None) -> 'KMeans':
		"""Fit the K-means model.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			Training instances to cluster.
		y : Ignored
			Not used, present for API consistency.
		sample_weight : array-like of shape (n_samples,), default=None
			The weights for each observation in X.

		Returns
		-------
		self : object
			Fitted estimator.
		"""
		# Validate and prepare data
		X = validate_data(self, X, dtype=[np.float64, np.float32], order='C')
		self._validate_params(X)

		# Handle sample weights
		if sample_weight is None:
			sample_weight = np.ones(X.shape[0], dtype=X.dtype)
		else:
			sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

		# Convert to JAX arrays
		X_jax = _to_jax_safe(X)
		weights_jax = _to_jax_safe(sample_weight)

		random_state = check_random_state(self.random_state)

		# Run multiple initializations
		best_inertia = np.inf
		best_labels = None
		best_centers = None
		best_n_iter = 0

		for init_run in range(self.n_init):
			if self.verbose:
				print(f"Initialization {init_run + 1}/{self.n_init}")

			labels, centers, inertia, n_iter = self._fit_single(X_jax, weights_jax, random_state)

			if inertia < best_inertia:
				best_inertia = inertia
				best_labels = labels
				best_centers = centers
				best_n_iter = n_iter

		# Store results
		self.cluster_centers_ = _to_numpy_safe(best_centers)
		self.labels_ = _to_numpy_safe(best_labels).astype(np.int32)
		self.inertia_ = best_inertia
		self.n_iter_ = best_n_iter

		return self

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""Predict cluster labels for samples in X.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			New data to predict.

		Returns
		-------
		labels : ndarray of shape (n_samples,)
			Index of the cluster each sample belongs to.
		"""
		check_is_fitted(self)
		X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)

		X_jax = _to_jax_safe(X)
		centers_jax = _to_jax_safe(self.cluster_centers_)

		labels_jax = _assign_labels_fast(X_jax, centers_jax)
		return _to_numpy_safe(labels_jax).astype(np.int32)

	def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None,
	                sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
		"""Fit the model and predict cluster labels.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			Training instances to cluster.
		y : Ignored
			Not used, present for API consistency.
		sample_weight : array-like of shape (n_samples,), default=None
			The weights for each observation in X.

		Returns
		-------
		labels : ndarray of shape (n_samples,)
			Index of the cluster each sample belongs to.
		"""
		return self.fit(X, sample_weight=sample_weight).labels_

	def transform(self, X: np.ndarray) -> np.ndarray:
		"""Transform X to a cluster-distance space.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			New data to transform.

		Returns
		-------
		X_new : ndarray of shape (n_samples, n_clusters)
			X transformed in cluster-distance space.
		"""
		check_is_fitted(self)
		X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)

		X_jax = _to_jax_safe(X)
		centers_jax = _to_jax_safe(self.cluster_centers_)

		distances_jax = jnp.sqrt(_squared_distances_vectorized(X_jax, centers_jax))
		return _to_numpy_safe(distances_jax)

	def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None,
	                  sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
		"""Fit the model and transform X to cluster-distance space.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			Training instances to cluster.
		y : Ignored
			Not used, present for API consistency.
		sample_weight : array-like of shape (n_samples,), default=None
			The weights for each observation in X.

		Returns
		-------
		X_new : ndarray of shape (n_samples, n_clusters)
			X transformed in cluster-distance space.
		"""
		return self.fit(X, sample_weight=sample_weight).transform(X)

	def score(self, X: np.ndarray, y: Optional[np.ndarray] = None, sample_weight: Optional[np.ndarray] = None) -> float:
		"""Opposite of the value of X on the K-means objective.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			New data.
		y : Ignored
			Not used, present for API consistency.
		sample_weight : array-like of shape (n_samples,), default=None
			The weights for each observation in X.

		Returns
		-------
		score : float
			Opposite of the value of X on the K-means objective.
		"""
		check_is_fitted(self)
		X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)

		if sample_weight is None:
			sample_weight = np.ones(X.shape[0], dtype=X.dtype)
		else:
			sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

		X_jax = _to_jax_safe(X)
		centers_jax = _to_jax_safe(self.cluster_centers_)
		weights_jax = _to_jax_safe(sample_weight)

		labels_jax = _assign_labels_fast(X_jax, centers_jax)
		inertia = _weighted_inertia_fast(X_jax, centers_jax, labels_jax, weights_jax)

		return -float(inertia)


# Optimized functional API

def k_means(X: np.ndarray,
            n_clusters: int,
            *,
            sample_weight: Optional[np.ndarray] = None,
            init: Union[str, np.ndarray] = 'k-means++',
            n_init: int = 10,
            max_iter: int = 300,
            verbose: bool = False,
            tol: float = 1e-4,
            random_state: Optional[int] = None,
            return_n_iter: bool = False) -> Union[
	Tuple[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray, float, int]]:
	"""Optimized K-means clustering algorithm using JAX.

	Parameters
	----------
	X : array-like of shape (n_samples, n_features)
		The observations to cluster.
	n_clusters : int
		The number of clusters to form.
	sample_weight : array-like of shape (n_samples,), default=None
		The weights for each observation in X.
	init : {'k-means++', 'random'} or ndarray, default='k-means++'
		Method for initialization.
	n_init : int, default=10
		Number of time the k-means algorithm will be run.
	max_iter : int, default=300
		Maximum number of iterations.
	verbose : bool, default=False
		Verbosity mode.
	tol : float, default=1e-4
		Relative tolerance for convergence.
	random_state : int, RandomState instance or None, default=None
		Random state for reproducibility.
	return_n_iter : bool, default=False
		Whether to return the number of iterations.

	Returns
	-------
	centroid : ndarray of shape (n_clusters, n_features)
		Centroids found at the last iteration.
	label : ndarray of shape (n_samples,)
		label[i] is the code or index of the centroid the i'th observation is closest to.
	inertia : float
		The final value of the inertia criterion.
	best_n_iter : int
		Number of iterations corresponding to the best results.
		Returned only if `return_n_iter` is set to True.
	"""
	estimator = KMeans(
		n_clusters=n_clusters,
		init=init,
		n_init=n_init,
		max_iter=max_iter,
		verbose=verbose,
		tol=tol,
		random_state=random_state
	).fit(X, sample_weight=sample_weight)

	if return_n_iter:
		return estimator.cluster_centers_, estimator.labels_, estimator.inertia_, estimator.n_iter_
	else:
		return estimator.cluster_centers_, estimator.labels_, estimator.inertia_


def kmeans_plusplus(X: np.ndarray,
                    n_clusters: int,
                    *,
                    sample_weight: Optional[np.ndarray] = None,
                    random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
	"""Optimized K-means++ initialization using JAX.

	Parameters
	----------
	X : array-like of shape (n_samples, n_features)
		The data to pick seeds for.
	n_clusters : int
		The number of centroids to initialize.
	sample_weight : array-like of shape (n_samples,), default=None
		The weights for each observation in X.
	random_state : int, RandomState instance or None, default=None
		Random state for reproducibility.

	Returns
	-------
	centers : ndarray of shape (n_clusters, n_features)
		The initial centers for k-means.
	indices : ndarray of shape (n_clusters,)
		The index location of the chosen centers in the data array X.
	"""
	X = check_array(X, dtype=[np.float64, np.float32])

	if sample_weight is None:
		sample_weight = np.ones(X.shape[0], dtype=X.dtype)
	else:
		sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

	random_state = check_random_state(random_state)
	key = random.PRNGKey(random_state.randint(0, 2 ** 31))

	X_jax = _to_jax_safe(X)
	weights_jax = _to_jax_safe(sample_weight)

	centers_jax = _kmeans_plusplus_init_optimized(X_jax, weights_jax, key, n_clusters)
	centers = _to_numpy_safe(centers_jax)

	# Find indices of chosen centers
	distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
	indices = np.argmin(distances, axis=0)

	return centers, indices