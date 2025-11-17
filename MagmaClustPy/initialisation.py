"""
This script will feature functions used to initialise elements of the MagmaClustPy model, including:
* initial hyperparameters of kernels
* the prior mean
* initial mixture of mean processes
* ...
"""

import jax.numpy as jnp
from MagmaClustPy.kmeans import k_means

# TODO: init_kernels

# TODO: init_prior_mean


def k_means_init(padded_outputs, k, distinct_hp=False):
	"""
	Compute the initial assignment of outputs between k clusters using k_means and a naive dimensionality reduction based on task statistics (min, mean, max)

	:param padded_outputs: the outputs from each task, (jnp.array, shape=(T, Max_N))
	:param k: the number of clusters (int)
	:param distinct_hp: whether distinct hyperparameters are used (bool, default=False)
	:return: the initial mixture as a jnp.array of shape (T,)
	"""
	# Compute statistics
	features = jnp.stack([
		jnp.nanmin(padded_outputs, axis=0),  # Min
		jnp.nanmean(padded_outputs, axis=0),  # Mean
		jnp.nanmax(padded_outputs, axis=0)  # Max
	], axis=-1).squeeze()

	if distinct_hp:
		features = jnp.concatenate([
			features,
			jnp.nanvar(padded_outputs, axis=0, ddof=1).reshape(-1, 1)  # Variance
		], axis=-1)

	# Run k-means
	_, labels, _= k_means(features, n_clusters=k, n_init=10, max_iter=100)

	return jnp.array(labels)
