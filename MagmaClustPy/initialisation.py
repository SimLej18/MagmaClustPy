"""
This script will feature functions used to initialise elements of the MagmaClustPy model, including:
* initial hyperparameters of kernels
* the prior mean
* initial mixture of mean processes
* ...
"""
from functools import partial

import jax.numpy as jnp
from jax import jit

from MagmaClustPy.kmeans import k_means

# TODO: init_kernels

# TODO: init_prior_mean


@partial(jit, static_argnums=(1, 2))
def k_means_init(padded_outputs, k, distinct_hp=False):
	"""
	Compute the initial assignment of outputs between k clusters using k_means and a naive dimensionality reduction based on task statistics (min, mean, max)

	:param padded_outputs: the outputs from each task, (jnp.array, shape=(T, Max_N))
	:param k: the number of clusters (int)
	:param distinct_hp: whether distinct hyperparameters are used (bool, default=False)
	:return: the initial mixture as a one-hot encoded array (jnp.array, shape=(k, T))
	"""
	outputs = padded_outputs.squeeze().T

	# Compute statistics
	features = jnp.stack([
		jnp.nanmin(outputs, axis=0),  # Min
		jnp.nanmean(outputs, axis=0),  # Mean
		jnp.nanmax(outputs, axis=0)  # Max
	], axis=-1).squeeze()

	if distinct_hp:
		features = jnp.concatenate([
			features,
			jnp.nanvar(outputs, axis=0, ddof=1).reshape(-1, 1)  # Variance
		], axis=-1)

	# Run k-means
	_, labels, _= k_means(features, n_clusters=k, n_init=10, max_iter=100)

	# One-hot encode labels
	labels = jnp.eye(k)[labels]

	return labels.T
