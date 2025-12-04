from typing import Tuple
from functools import partial

from jax import jit
from jax import numpy as jnp

from kernax import AbstractKernel
from MagmaClustPy.linalg import cho_factor, cho_solve, map_to_full_matrix_batch, map_to_full_array_batch


@jit
def hyperpost_shared_input(outputs: jnp.ndarray, prior_mean: jnp.ndarray, mean_cov_u: jnp.ndarray,
                           mean_prec: jnp.ndarray, task_covs: jnp.ndarray, mixture_coeffs: jnp.ndarray) \
		-> Tuple[jnp.ndarray, jnp.ndarray]:
	eye = jnp.eye(task_covs.shape[-1])

	# Compute task covariance, its Cholesky factor and its inverse aka precision
	task_covs_u = cho_factor(task_covs)
	task_prec = cho_solve(task_covs_u, eye)

	if task_prec.ndim == 2:
		# Shared inputs and shared HPs, all covs are the same, so we only have only one
		post_prec_u = cho_factor(mean_prec + mixture_coeffs.sum() * task_prec)
	else:
		# task_prec has a batch dimension, we have distinct HPs
		post_prec_u = cho_factor(mean_prec + (mixture_coeffs[:, None, None] * task_prec).sum(axis=0))
	post_cov = cho_solve(post_prec_u, eye)

	# Compute posterior mean
	weighted_prior_mean = cho_solve(mean_cov_u, prior_mean)

	weighted_tasks = (mixture_coeffs[:, None] * cho_solve(task_covs_u, outputs)).sum(axis=0)

	post_mean = cho_solve(post_prec_u, weighted_prior_mean + weighted_tasks)

	return post_mean, post_cov


@jit
def hyperpost_distinct_input(outputs: jnp.ndarray, mappings: jnp.ndarray, all_inputs: jnp.ndarray,
                             prior_mean: jnp.ndarray, mean_cov_u: jnp.ndarray, mean_cov_inv: jnp.ndarray,
                             task_covs: jnp.ndarray, mixture_coeffs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
	"""
	computes the hyperpost on distinct inputs

	task_covs: (M, N, N), batch of unaligned covariances
	"""
	small_eye = jnp.eye(task_covs.shape[-1])

	# task_covs is padded with NaNs. Replace them by their corresponding identity rows/cols
	eyed_task_covs = jnp.where(jnp.isnan(task_covs), small_eye, task_covs)
	# Posterior covariance
	task_covs_U = cho_factor(eyed_task_covs)
	task_covs_inv = cho_solve(task_covs_U, small_eye)
	task_covs_inv -= jnp.where(jnp.isnan(task_covs), small_eye, 0)  # Correction on the diagonal
	task_covs_inv = map_to_full_matrix_batch(task_covs_inv, all_inputs, mappings)
	task_cov_inv = (mixture_coeffs[:, None, None] * jnp.nan_to_num(task_covs_inv)).sum(axis=0)

	post_cov_inv = cho_factor(mean_cov_inv + task_cov_inv)
	post_cov = cho_solve(post_cov_inv, jnp.eye(mean_cov_u.shape[-1]))

	# Posterior mean
	weighted_prior_mean = cho_solve(mean_cov_u, prior_mean)
	mapped_outputs = jnp.nan_to_num(map_to_full_array_batch(outputs, all_inputs, mappings))
	padded_task_covs_U = map_to_full_matrix_batch(task_covs_U, all_inputs, mappings)
	eyed_task_covs_U = jnp.where(jnp.isnan(padded_task_covs_U), jnp.eye(all_inputs.shape[0]), padded_task_covs_U)
	weighted_tasks = (mixture_coeffs[:, None] * cho_solve(eyed_task_covs_U, mapped_outputs)).sum(axis=0)

	post_mean = cho_solve(post_cov_inv, weighted_prior_mean + weighted_tasks)

	return post_mean, post_cov


# General function
@partial(jit, static_argnums=(7, 8))
def hyperpost(inputs: jnp.ndarray, outputs: jnp.ndarray, mappings: jnp.ndarray, all_inputs: jnp.ndarray,
              prior_mean: jnp.ndarray, mean_kernel: AbstractKernel, task_kernel: AbstractKernel,
              shared_input: bool, shared_hp: bool, mixture_coeffs: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
	"""
	Computes the posterior mean and covariance of a Magma GP given the inputs, outputs, mappings, prior mean and kernels.

	:param inputs: Inputs of every point, for every task, padded with NaNs. Shape (T, Max_N_i, I)
	:param outputs: Outputs of every point, for every task, padded with NaNs. Shape (T, Max_N_i, O)
	:param mappings: Indices of every input in the all_inputs array, padded with len(all_inputs). Shape (T, Max_N_i)
	:param all_inputs: all distinct inputs. Shape (N, I)
	:param prior_mean: prior mean over all_inputs or grid if provided. Shape (N,) or (G,), or scalar if constant
	across the domain.
	:param mean_kernel: Kernel to be used to compute the mean covariance.
	:param task_kernel: Kernel to be used to compute the task covariance.
	:param mixture_coeffs: optional mixture coefficients for every task. Shape(T,). Default is None.
	Shape (G, I), when provided it is merged with all_inputs to keep information in the model.

	:return: a 2-tuple of the posterior mean and covariance
	"""
	# In multi-output, we want to flatten the outputs.
	# The user should provide a specific Kernel to compute a cross-covariance with the right shape too
	outputs = outputs.reshape(outputs.shape[0], -1)

	# If mixture coefficients are not provided, assume uniform weights
	if mixture_coeffs is None:
		mixture_coeffs = jnp.ones((inputs.shape[0],))

	if prior_mean.ndim == 0:
		prior_mean = jnp.broadcast_to(prior_mean, (len(all_inputs),))

	# Compute mean covariance and its Cholesky factor
	mean_cov = mean_kernel(all_inputs, all_inputs)
	mean_cov_u = cho_factor(mean_cov)
	mean_cov_inv = cho_solve(mean_cov_u, jnp.eye(all_inputs.shape[0]))

	if shared_input:
		if shared_hp:
			# FIXME: this assumes that the outer-most kernel is a BatchKernel
			task_covs = task_kernel.inner_kernel(all_inputs)  # Shape: (Max_Ni, Max_Ni)
		else:
			# TODO: possible optimisation
			#  Rather than computing the kernel on inputs, compute it on all_inputs, broadcast to (T, Max_Ni, Max_Ni),
			#  apply a mask using mappings and then invert. This may avoid a lot of kernel calls.
			task_covs = task_kernel(inputs)  # Shape: (T, Max_Ni, Max_Ni)

		return hyperpost_shared_input(outputs, prior_mean, mean_cov_u, mean_cov_inv, task_covs, mixture_coeffs)

	else:  # No shared input: we have to pad and mapping
		task_covs = task_kernel(inputs)  # Shape: (T, Max_Ni, Max_Ni)

		return hyperpost_distinct_input(outputs, mappings, all_inputs, prior_mean, mean_cov_u, mean_cov_inv,
		                                task_covs, mixture_coeffs)
