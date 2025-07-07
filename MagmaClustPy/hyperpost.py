from MagmaClustPy.linalg import map_to_full_matrix_batch, map_to_full_array_batch

from jax import jit
from jax import numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from jax.tree_util import tree_flatten


@jit
def hyperpost_shared_input_shared_hp(outputs, prior_mean, mean_cov_u, mean_cov_inv, task_cov, inputs_to_grid=None,
                                     nugget=jnp.array(1e-10)):
	eye = jnp.eye(task_cov.shape[-1])

	# Compute task covariance and its Cholesky factor
	task_cov_u, _ = cho_factor(task_cov + eye * nugget)
	task_cov_inv = cho_solve((task_cov_u, False), eye)

	if inputs_to_grid is not None:
		task_cov_inv = jnp.zeros_like(mean_cov_inv).at[jnp.ix_(inputs_to_grid, inputs_to_grid)].set(task_cov_inv)

	# All tasks share same inputs and hyperparameters, so their inverse covariances are the same, and we can compute
	# one then multiply rather than compute all then sum
	post_cov_inv, _ = cho_factor(mean_cov_inv + len(outputs) * task_cov_inv, )
	post_cov = cho_solve((post_cov_inv, False), eye)

	# Compute posterior mean
	weighted_prior_mean = cho_solve((mean_cov_u, False), prior_mean)
	weighted_tasks = cho_solve((task_cov_u, False), outputs.sum(axis=0))

	if inputs_to_grid is not None:
		weighted_tasks = jnp.zeros_like(prior_mean).at[inputs_to_grid].set(weighted_tasks)

	post_mean = cho_solve((post_cov_inv, False), weighted_prior_mean + weighted_tasks)

	return post_mean, post_cov


@jit
def hyperpost_shared_input_distinct_hp(outputs, prior_mean, mean_cov_u, mean_cov_inv, task_covs, inputs_to_grid=None,
                                       nugget=jnp.array(1e-10)):
	eye = jnp.broadcast_to(jnp.eye(task_covs.shape[-1]), task_covs.shape)

	# Compute task covariance and its Cholesky factor
	task_covs_u, _ = cho_factor(task_covs + eye * nugget)
	task_cov_inv = cho_solve((task_covs_u, False), eye)

	task_cov_inv = task_cov_inv.sum(axis=0)

	if inputs_to_grid is not None:
		task_cov_inv = jnp.zeros_like(mean_cov_inv).at[jnp.ix_(inputs_to_grid, inputs_to_grid)].set(task_cov_inv)

	post_cov_inv, _ = cho_factor(mean_cov_inv + task_cov_inv)
	post_cov = cho_solve((post_cov_inv, False), eye[0])

	# Compute posterior mean
	weighted_prior_mean = cho_solve((mean_cov_u, False), prior_mean)
	# weighted_tasks = vmap(lambda L, o: cho_solve((L, True), o))(task_covs_L, outputs).sum(axis=0)
	weighted_tasks = cho_solve((task_covs_u, False), outputs).sum(axis=0)

	if inputs_to_grid is not None:
		weighted_tasks = jnp.zeros_like(prior_mean).at[inputs_to_grid].set(weighted_tasks)

	post_mean = cho_solve((post_cov_inv, False), weighted_prior_mean + weighted_tasks)

	return post_mean, post_cov


@jit
def hyperpost_distinct_input(outputs, mappings, prior_mean, mean_cov_u, mean_cov_inv, task_covs, all_inputs,
                             inputs_to_grid=None,
                             nugget=jnp.array(1e-10)):
	"""
	computes the hyperpost on distinct inputs

	task_covs: (M, N, N), batch of unaligned covariances
	"""
	small_eye = jnp.broadcast_to(jnp.eye(task_covs.shape[-1]), task_covs.shape)

	# task_covs is padded with NaNs. Replace them by their corresponding identity rows/cols
	eyed_task_covs = jnp.where(jnp.isnan(task_covs), small_eye, task_covs)

	# Posterior covariance
	task_covs_U, _ = cho_factor(eyed_task_covs + small_eye * nugget)
	task_covs_inv = cho_solve((task_covs_U, False), small_eye)
	task_covs_inv -= jnp.where(jnp.isnan(task_covs), small_eye, 0)  # Correction on the diagonal
	task_covs_inv = map_to_full_matrix_batch(task_covs_inv, all_inputs, mappings)
	task_cov_inv = jnp.nan_to_num(task_covs_inv).sum(axis=0)

	if inputs_to_grid is not None:
		task_cov_inv = jnp.zeros_like(mean_cov_inv).at[jnp.ix_(inputs_to_grid, inputs_to_grid)].set(task_cov_inv)

	post_cov_inv, _ = cho_factor(mean_cov_inv + task_cov_inv)
	post_cov = cho_solve((post_cov_inv, False), jnp.eye(mean_cov_u.shape[-1]))

	# Posterior mean
	weighted_prior_mean = cho_solve((mean_cov_u, False), prior_mean)
	mapped_outputs = jnp.nan_to_num(map_to_full_array_batch(outputs, all_inputs, mappings))
	padded_task_covs_U = map_to_full_matrix_batch(task_covs_U, all_inputs, mappings)
	eyed_task_covs_U = jnp.where(jnp.isnan(padded_task_covs_U), jnp.eye(all_inputs.shape[-1]), padded_task_covs_U)
	weighted_tasks = cho_solve((eyed_task_covs_U, False), mapped_outputs).sum(axis=0)

	if inputs_to_grid is not None:
		weighted_tasks = jnp.zeros_like(prior_mean).at[inputs_to_grid].set(weighted_tasks)

	post_mean = cho_solve((post_cov_inv, False), weighted_prior_mean + weighted_tasks)

	return post_mean, post_cov


# General function
def hyperpost(inputs, outputs, mappings, prior_mean, mean_kernel, task_kernel, all_inputs=None, grid=None,
              nugget=jnp.array(1e-10)):
	"""
	Computes the posterior mean and covariance of a Magma GP given the inputs, outputs, mappings, prior mean and kernels.

	:param inputs: Inputs of every point, for every task, padded with NaNs. Shape (T, Max_N_i, I)
	:param outputs: Outputs of every point, for every task, padded with NaNs. Shape (T, Max_N_i, O)
	:param mappings: Indices of every input in the all_inputs array, padded with len(all_inputs). Shape (T, Max_N_i)
	:param prior_mean: prior mean over all_inputs or grid if provided. Shape (N,) or (G,), or scalar if constant
	across the domain.
	:param mean_kernel: Kernel to be used to compute the mean covariance.
	:param task_kernel: Kernel to be used to compute the task covariance.
	:param all_inputs: all distinct inputs. If not provided, it will be computed from the inputs. Shape (N, I)
	:param grid: the grid on which the GP is defined. If not provided, the GP is defined on all distinct inputs.
	Shape (G, I)
	:param nugget: nugget term to ensure numerical stability. Default is 1e-10

	:return: a 2-tuple of the posterior mean and covariance
	"""
	# TODO: add a dimension for clusters in the returned hyperpost
	# In multi-output, we want to flatten the outputs.
	# The user should provide a specific Kernel to compute a cross-covariance with the right shape too
	outputs = outputs.reshape(outputs.shape[0], -1)

	shared_hp = all([hp.ndim == 0 for hp in tree_flatten(task_kernel)[0]])

	# Merge inputs and grid to create all_inputs
	# TODO: maybe we can assume the user will always provide all_inputs, as it's given by the preprocessing
	if all_inputs is None:
		all_inputs = jnp.sort(jnp.unique(inputs, axis=0))

	shared_input = len(inputs[0]) == len(all_inputs)

	if grid is None:
		grid = all_inputs
		inputs_to_grid = None
	else:
		grid = jnp.sort(jnp.unique(jnp.concatenate([all_inputs, grid])))
		inputs_to_grid = jnp.searchsorted(grid, all_inputs)
		shared_input = False  # We need to pad the cov matrices to compute on the full grid

	if prior_mean.ndim == 0:
		prior_mean = jnp.broadcast_to(prior_mean, (len(grid),))

	# Numerical stability terms
	eye = jnp.eye(grid.shape[0])

	# Compute mean covariance and its Cholesky factor
	mean_cov = mean_kernel(grid, grid)
	mean_cov_u, _ = cho_factor(mean_cov + eye * nugget)
	mean_cov_inv = cho_solve((mean_cov_u, False), eye)

	if shared_input:
		if shared_hp:
			task_cov = task_kernel(grid)  # Shape: (N, N)
			return hyperpost_shared_input_shared_hp(outputs, prior_mean, mean_cov_u, mean_cov_inv, task_cov,
			                                        inputs_to_grid, nugget)

		else:  # distinct HPs, we have to compute every task covariance but no padding is required
			task_covs = task_kernel(inputs)  # Shape: (M, N, N)
			return hyperpost_shared_input_distinct_hp(outputs, prior_mean, mean_cov_u, mean_cov_inv, task_covs,
			                                          inputs_to_grid, nugget)

	else:  # No shared input: we have to pad and mapping
		# task_covs = task_kernel(jnp.broadcast_to(all_inputs, (len(inputs), len(all_inputs))))
		task_covs = task_kernel(inputs)
		return hyperpost_distinct_input(outputs, mappings, prior_mean, mean_cov_u, mean_cov_inv, task_covs, all_inputs,
		                                inputs_to_grid, nugget)
