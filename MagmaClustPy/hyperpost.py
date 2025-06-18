from jax import jit
from jax import numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from jax.tree_util import tree_flatten


# JITed, specialized functions
@jit
def hyperpost_common_input_common_hp(outputs, prior_mean, mean_cov_u, mean_cov_inv, task_cov, inputs_to_grid=None,
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
def hyperpost_common_input_distinct_hp(outputs, prior_mean, mean_cov_u, mean_cov_inv, task_covs, inputs_to_grid=None,
                                       nugget=jnp.array(1e-10)):
	eye = jnp.broadcast_to(jnp.eye(task_covs.shape[-1]), task_covs.shape)

	# Compute task covariance and its Cholesky factor
	# task_covs_L = vmap(lambda x: cho_factor(x + eye * nugget, lower=True)[0])(task_covs)
	task_covs_u, _ = cho_factor(task_covs + eye * nugget)
	# task_cov_inv = vmap(lambda L: cho_solve((L, True), eye))(task_covs_L).sum(axis=0)
	task_cov_inv = cho_solve((task_covs_u, False), eye)

	if inputs_to_grid is not None:
		task_cov_inv = jnp.zeros_like(mean_cov_inv).at[jnp.ix_(inputs_to_grid, inputs_to_grid)].set(task_cov_inv)

	task_cov_inv = task_cov_inv.sum(axis=0)

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
def hyperpost_distinct_input(outputs, masks, prior_mean, mean_cov_u, mean_cov_inv, task_covs, inputs_to_grid=None,
                             nugget=jnp.array(1e-10)):
	"""
	computes the hyperpost on distinct inputs

	task_covs: (M, N, N), batch of unaligned covariances
	"""
	small_eye = jnp.broadcast_to(jnp.eye(task_covs.shape[-1]), task_covs.shape)
	big_eye = jnp.eye(mean_cov_u.shape[-1])

	# task_covs is padded with NaNs. Replace them by their corresponding identity rows/cols
	masks_2D = masks[:, :, None] & masks[:, None, :]
	task_covs = jnp.where(masks_2D, task_covs, small_eye)

	# Posterior covariance
	# task_covs_L = vmap(lambda x: cho_factor(x + small_eye * nugget)[0])(task_covs)
	task_covs_U, _ = cho_factor(task_covs + small_eye * nugget)
	# task_covs_inv = vmap(lambda L: cho_solve((L, False), small_eye))(task_covs_L)
	task_covs_inv = cho_solve((task_covs_U, False), small_eye)
	task_covs_inv -= jnp.where(masks_2D, 0, small_eye)  # Correction on the diagonal
	task_cov_inv = task_covs_inv.sum(axis=0)

	if inputs_to_grid is not None:
		task_cov_inv = jnp.zeros_like(mean_cov_inv).at[jnp.ix_(inputs_to_grid, inputs_to_grid)].set(task_cov_inv)

	post_cov_inv, _ = cho_factor(mean_cov_inv + task_cov_inv)
	post_cov = cho_solve((post_cov_inv, False), big_eye)

	# Posterior mean
	weighted_prior_mean = cho_solve((mean_cov_u, False), prior_mean)
	outputs = jnp.where(masks, outputs, 0)
	# weighted_tasks = vmap(lambda L, o: cho_solve((L, False), o))(task_covs_L, outputs).sum(axis=0)
	weighted_tasks = cho_solve((task_covs_U, False), outputs).sum(axis=0)

	if inputs_to_grid is not None:
		weighted_tasks = jnp.zeros_like(prior_mean).at[inputs_to_grid].set(weighted_tasks)

	post_mean = cho_solve((post_cov_inv, False), weighted_prior_mean + weighted_tasks)

	return post_mean, post_cov


# General function
def hyperpost(inputs, outputs, masks, prior_mean, mean_kernel, task_kernel, all_inputs=None, grid=None,
              nugget=jnp.array(1e-10)):
	"""
	Computes the posterior mean and covariance of a Magma GP given the inputs, outputs, masks, prior mean and kernels.

	:param inputs: the preprocessed (padded and aligned) inputs
	:param outputs: the preprocessed outputs
	:param masks: the masks indicating which inputs are valid
	:param prior_mean: the prior mean, as a scalar or a vector of shape (N, ), where N is the length of the union of all
	inputs and the grid
	:param mean_kernel: kernel of the mean process, with hyperparameters loaded as attributes
	:param task_kernel: kernel of the task process, with hyperparameters loaded as attributes
	:param all_inputs: all distinct inputs. If not provided, it will be computed from the inputs
	:param grid: the grid on which the GP is defined. If not provided, the GP is defined on all distinct inputs
	:param nugget: nugget term to ensure numerical stability. Default is 1e-10
	:return: a 2-tuple of the posterior mean and covariance
	"""
	common_input = jnp.all(masks)
	common_hp = all([hp.ndim == 0 for hp in tree_flatten(task_kernel)[0]])

	# Merge inputs and grid to create all_inputs
	if all_inputs is None:
		if common_input:
			all_inputs = inputs[0]
		else:
			all_inputs = jnp.sort(jnp.unique(inputs.flatten()))

	if grid is None:
		grid = all_inputs
		inputs_to_grid = None
	else:
		grid = jnp.sort(jnp.unique(jnp.concatenate([all_inputs, grid])))
		inputs_to_grid = jnp.searchsorted(grid, all_inputs)
		common_input = False  # We need to pad the cov matrices to compute on the full grid

	if prior_mean.ndim == 0:
		prior_mean = jnp.broadcast_to(prior_mean, (len(grid),))

	# Numerical stability terms
	eye = jnp.eye(grid.shape[0])

	# Compute mean covariance and its Cholesky factor
	mean_cov = mean_kernel(grid, grid)
	mean_cov_u, _ = cho_factor(mean_cov + eye * nugget)
	mean_cov_inv = cho_solve((mean_cov_u, False), eye)

	if common_input:
		if common_hp:
			task_cov = task_kernel(grid)  # Shape: (N, N)
			return hyperpost_common_input_common_hp(outputs, prior_mean, mean_cov_u, mean_cov_inv, task_cov,
			                                        inputs_to_grid, nugget)

		else:  # distinct HPs, we have to compute every task covariance but no padding is required
			task_covs = task_kernel(inputs)  # Shape: (M, N, N)
			return hyperpost_common_input_distinct_hp(outputs, prior_mean, mean_cov_u, mean_cov_inv, task_covs,
			                                          inputs_to_grid, nugget)

	else:  # No common input: we have to pad and mask
		# task_covs = task_kernel(jnp.broadcast_to(all_inputs, (len(inputs), len(all_inputs))))
		task_covs = task_kernel(inputs)
		return hyperpost_distinct_input(outputs, masks, prior_mean, mean_cov_u, mean_cov_inv, task_covs, inputs_to_grid,
		                                nugget)
