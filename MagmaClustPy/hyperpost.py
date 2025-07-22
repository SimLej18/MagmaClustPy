from typing import Tuple, Optional

from jax import jit, vmap
from jax import numpy as jnp

from Kernax import AbstractKernel
from MagmaClustPy.linalg import cho_factor, cho_solve, map_to_full_matrix_batch, map_to_full_array_batch, lexicographic_sort, compute_mapping


@jit
def hyperpost_shared_input(outputs: jnp.ndarray, prior_mean: jnp.ndarray, mean_cov_u: jnp.ndarray,
                           mean_prec: jnp.ndarray, task_covs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
	eye = jnp.eye(task_covs.shape[-1])

	# Compute task covariance, its Cholesky factor and its inverse aka precision
	task_covs_u = cho_factor(task_covs)
	task_prec = cho_solve(task_covs_u, eye)

	if task_prec.ndim == 2:
		# Shared inputs and shared HPs, all covs are the same, so we only have only one
		post_prec_u = cho_factor(mean_prec + len(outputs) * task_prec)
	else:
		# task_prec has a batch dimension, we have distinct HPs
		post_prec_u = cho_factor(mean_prec + task_prec.sum(axis=0))
	post_cov = cho_solve(post_prec_u, eye)

	# Compute posterior mean
	weighted_prior_mean = cho_solve(mean_cov_u, prior_mean)

	weighted_tasks = cho_solve(task_covs_u, outputs).sum(axis=0)

	post_mean = cho_solve(post_prec_u, weighted_prior_mean + weighted_tasks)

	return post_mean, post_cov


@jit
def hyperpost_distinct_input(outputs: jnp.ndarray, mappings: jnp.ndarray, all_inputs: jnp.ndarray,
                             prior_mean: jnp.ndarray, mean_cov_u: jnp.ndarray, mean_cov_inv: jnp.ndarray,
                             task_covs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
	task_cov_inv = jnp.nan_to_num(task_covs_inv).sum(axis=0)

	post_cov_inv = cho_factor(mean_cov_inv + task_cov_inv)
	post_cov = cho_solve(post_cov_inv, jnp.eye(mean_cov_u.shape[-1]))

	# Posterior mean
	weighted_prior_mean = cho_solve(mean_cov_u, prior_mean)
	mapped_outputs = jnp.nan_to_num(map_to_full_array_batch(outputs, all_inputs, mappings))
	padded_task_covs_U = map_to_full_matrix_batch(task_covs_U, all_inputs, mappings)
	eyed_task_covs_U = jnp.where(jnp.isnan(padded_task_covs_U), jnp.eye(all_inputs.shape[0]), padded_task_covs_U)
	weighted_tasks = cho_solve(eyed_task_covs_U, mapped_outputs).sum(axis=0)

	post_mean = cho_solve(post_cov_inv, weighted_prior_mean + weighted_tasks)

	return post_mean, post_cov


# General function
@jit
def hyperpost(inputs: jnp.ndarray, outputs: jnp.ndarray, mappings: jnp.ndarray, all_inputs: jnp.ndarray,
              prior_mean: jnp.ndarray, mean_kernel: AbstractKernel, task_kernel: AbstractKernel, grid: Optional[jnp.ndarray] = None) \
		-> Tuple[jnp.ndarray, jnp.ndarray]:
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
	:param grid: the grid on which the GP is defined. If not provided, the GP is defined on all distinct inputs.
	Shape (G, I), when provided it is merged with all_inputs to keep information in the model.

	:return: a 2-tuple of the posterior mean and covariance
	"""
	# TODO: add a dimension for clusters in the returned hyperpost
	# In multi-output, we want to flatten the outputs.
	# The user should provide a specific Kernel to compute a cross-covariance with the right shape too
	outputs = outputs.reshape(outputs.shape[0], -1)

	# Merge inputs and grid to create all_inputs
	shared_input = len(inputs[0]) == len(all_inputs)
	shared_hp = not task_kernel.has_distinct_hyperparameters(inputs.shape[0])

	if grid is None:
		grid = all_inputs
	else:
		grid = lexicographic_sort(jnp.concatenate([all_inputs, grid]))
		# FIXME: concatenating all_inputs and grid might introduce duplicates,
		#  but we can't use jnp.unique in a jitted function without knowing the new dimension in advance.
		#  It's unclear if those duplicate points might introduce numerical problems later.
		mappings = vmap(compute_mapping, in_axes=(None, 0))(grid, inputs)
		shared_input = False  # We need to pad the cov matrices to compute on the full grid

	if prior_mean.ndim == 0:
		prior_mean = jnp.broadcast_to(prior_mean, (len(grid),))

	# Compute mean covariance and its Cholesky factor
	mean_cov = mean_kernel(grid, grid)
	mean_cov_u = cho_factor(mean_cov)
	mean_cov_inv = cho_solve(mean_cov_u, jnp.eye(grid.shape[0]))

	if shared_input:
		if shared_hp:
			task_covs = task_kernel(grid)  # Shape: (Max_Ni, Max_Ni)
		else:
			task_covs = task_kernel(inputs)  # Shape: (T, Max_Ni, Max_Ni)

		return hyperpost_shared_input(outputs, prior_mean, mean_cov_u, mean_cov_inv, task_covs)

	else:  # No shared input: we have to pad and mapping
		task_covs = task_kernel(inputs)  # Shape: (T, Max_Ni, Max_Ni)

		return hyperpost_distinct_input(outputs, mappings, grid, prior_mean, mean_cov_u, mean_cov_inv,
		                                task_covs)
