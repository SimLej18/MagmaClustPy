from typing import Optional
from functools import partial

import jax.numpy as jnp
from jax import jit, vmap, Array
from jax.scipy.stats.multivariate_normal import logpdf
from kernax import AbstractKernel

from MagmaClustPy.linalg import extract_from_full_array, extract_from_full_matrix, solve_right_cholesky


@jit
def magma_nll(kernel: AbstractKernel, inputs: Array, outputs: Array, mean: Array,
                         mean_process_cov: Array, mappings: Optional[Array],
                         jitter: Optional[Array] = jnp.array(1e-10)) -> Array:
	"""
	Computes the Magma Negative Log Likelihood for a given sequence of points and covariance kernel, relative to a
	posterior mean and covariance.

	:param kernel: A kernel to compute the covariance on inputs
	:param inputs: The inputs from one task. Shape: (Max_N_i, D)
	:param outputs: The outputs from one task, corresponding to the inputs. Shape: (Max_N_i, D)
	:param mean: The posterior mean to which the outputs are compared. Shape (N, D)
	:param mean_process_cov: The posterior covariance matrix to which the task covariance is compared. Shape (N, N)
	:param mappings: The mappings from input points (in 0..Max_N_i) to full grid (0..N). Shape (Max_N_i, )
	:param jitter: The jitter to use. Scalar.

	:return: The negative log-likelihood with the magma correction term. Scalar.
	"""
	outputs = outputs.ravel()  # For multi-output, we want to flatten the outputs.
	mean = mean.ravel()  # As the goal of likelihood is to see if the mean is close to the outputs, we want to flatten
	# it too.

	jitter_matrix = jnp.eye(outputs.shape[0]) * jitter

	cov = kernel(inputs)

	eyed_cov = jnp.where(jnp.isnan(cov), jnp.eye(cov.shape[0]), cov)
	zeroed_outputs = jnp.nan_to_num(outputs)
	if mappings is not None:
		zeroed_mean = jnp.nan_to_num(extract_from_full_array(mean, outputs, mappings))
		eyed_mean_cov = jnp.where(jnp.isnan(cov), jnp.eye(cov.shape[0]),
		                          extract_from_full_matrix(mean_process_cov, outputs, mappings))
	else:
		zeroed_mean = jnp.nan_to_num(mean)
		eyed_mean_cov = jnp.where(jnp.isnan(cov), jnp.eye(cov.shape[0]), mean_process_cov)

	# Compute log-likelihood
	multiv_neg_log_lik = -logpdf(zeroed_outputs, zeroed_mean, eyed_cov + jitter_matrix)

	# Compute correction term
	correction = 0.5 * jnp.trace(solve_right_cholesky(eyed_cov, eyed_mean_cov, jitter=jitter))

	# Compute padding corrections
	# The logpdf is computed as:
	# -0.5 * (N * log(2 * pi) + log(det(cov)) + (outputs - mean).T @ inv(cov) @ (outputs - mean))
	# det(cov) and the Mahalanobis distance are not affected by our padding
	# We only have to correct for the -0.5 * N * log(2 * pi) term, as N is bigger with padding
	nll_pad_correction = 0.5 * jnp.log(2 * jnp.pi) * jnp.sum(jnp.isnan(outputs))

	# We also need to correct the correction term, as padding adds 1s to the diagonal and hence 1 to the trace
	corr_pad_correction = 0.5 * jnp.sum(jnp.isnan(outputs))

	return (multiv_neg_log_lik - nll_pad_correction) + (correction - corr_pad_correction)


@partial(jit, static_argnames=('shared_hp', 'cluster_hp'))
def compute_task_nlls(task_kernel: AbstractKernel, inputs: Array, outputs: Array, mappings: Array,
                      post_means: Array, post_covs: Array, shared_hp: bool, cluster_hp: bool, jitter: Array = 1e-5,):
	"""
	Computes the magma negative log-likelihood for each task of the dataset, relative to each cluster.
	:param task_kernel: Kernel to use to compute covariances of tasks
	:param inputs: shape (T, Max_N_i, I)
	:param outputs: shape (T, Max_N_i, O)
	:param mappings: shape (T, Max_N_i)
	:param post_means: shape (K, N)
	:param post_covs: shape (K, N, N)
	:param shared_hp: whether the kernel uses shared hyperparameters.
	:param cluster_hp: whether the kernel uses cluster hyperparameters.
	:param jitter: jitter term to ensure numerical stability.
	:return: likelihood values for each task relative to each cluster. Shape (K, T)
	"""
	task_batched_magma_nll = vmap(magma_nll, in_axes=(None if shared_hp else 0, 0, 0, None, None, 0, None))
	cluster_batched_magma_nll = vmap(
		lambda k, *args: task_batched_magma_nll(k.inner_kernel if cluster_hp else k, *args),
		in_axes=(0 if cluster_hp else None, None, None, 0, 0, None, None))
	return cluster_batched_magma_nll(task_kernel.inner_kernel,
	                                 inputs, outputs, post_means, post_covs, mappings, jitter)
