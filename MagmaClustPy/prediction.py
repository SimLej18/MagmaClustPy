import jax.numpy as jnp
from jax import jit, vmap

from MagmaClustPy.linalg import cho_factor
from jax.lax.linalg import triangular_solve


@jit
def predict_single_task(single_task_kernel, grid, post_cov_grid, post_mean_grid, padded_output_pred, mappings_pred_on_grid):
	gamma_on_grid = post_cov_grid + single_task_kernel(grid)

	post_mean_at_pred = jnp.where(~jnp.isnan(padded_output_pred), post_mean_grid[mappings_pred_on_grid], 0.)

	gamma_at_pred = gamma_on_grid[jnp.ix_(mappings_pred_on_grid, mappings_pred_on_grid)]

	# FIXME: there's a bug regarding indexing of gamma_crossed. Following comments are a first attempt to fix it.
	#  all_mappings = jnp.arange(0, len(grid))
	#  mappings_not_pred_on_grid = jnp.where(~jnp.isin(all_mappings, mappings_pred_on_grid), all_mappings, len(all_mappings))
	#  gamma_crossed = gamma_on_grid[mappings_pred_on_grid, mappings_not_pred_on_grid]
	gamma_crossed = gamma_on_grid[mappings_pred_on_grid, :]

	padding_mask = ~jnp.isnan(padded_output_pred)[:, None] & ~jnp.isnan(padded_output_pred)[None, :]
	padded_gamma_at_pred = jnp.where(padding_mask, gamma_at_pred, jnp.eye(len(gamma_at_pred)))
	padded_gamma_crossed = jnp.where(~jnp.isnan(padded_output_pred)[:, None], gamma_crossed, 0.)

	gamma_at_pred_U = cho_factor(padded_gamma_at_pred)
	z = triangular_solve(gamma_at_pred_U, padded_gamma_crossed.T).T
	y = triangular_solve(gamma_at_pred_U, jnp.nan_to_num(padded_output_pred) - post_mean_at_pred)

	pred_mean = post_mean_grid + (z.T @ y)
	pred_cov = gamma_on_grid - (z.T @ z)

	return pred_mean, pred_cov


@jit
def predict(post_mean_grid, post_cov_grid, padded_outputs_pred, mappings_pred_on_grid, grid, pred_task_kernel):
	# In multi-output, we want to flatten the outputs.
	# The user should provide a specific Kernel to compute a cross-covariance with the right shape too
	padded_outputs_pred = padded_outputs_pred.reshape(padded_outputs_pred.shape[0], -1)

	hp_vmap = pred_task_kernel.get_hp_vmap_in_axes(padded_outputs_pred.shape[0])

	return vmap(predict_single_task, in_axes=(hp_vmap, None, None, None, 0, 0))(pred_task_kernel, grid,
	                                                                            post_cov_grid, post_mean_grid,
	                                                                            padded_outputs_pred, mappings_pred_on_grid)
