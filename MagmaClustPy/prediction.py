from typing import Tuple
import jax.numpy as jnp
from jax import jit, vmap, Array
from jax.lax.linalg import triangular_solve

from MagmaClustPy.linalg import cho_factor


@jit
def predict_single_task(padded_inputs_pred, padded_outputs_pred, grid,
                        post_mean_pred, post_cov_pred, post_cov_crossed, post_mean_grid, post_cov_grid,
                        task_kernel) -> Tuple[Array, Array]:
	# FIXME: using `task_kernel.left_kernel` assumes a specific kernel structure with noise on the right.
	#  We are not sure that the user will build the pred kernel that way.
	post_mean_pred = jnp.where(~jnp.isnan(padded_outputs_pred), post_mean_pred, 0.)

	gamma_pred = post_cov_pred + task_kernel(padded_inputs_pred)
	gamma_grid = post_cov_grid + task_kernel.left_kernel(grid)  # No noise before
	#gamma_grid = post_cov_grid + task_kernel(grid)
	gamma_crossed = post_cov_crossed + task_kernel.left_kernel(padded_inputs_pred, grid)

	padding_mask = ~jnp.isnan(padded_outputs_pred)[:, None] & ~jnp.isnan(padded_outputs_pred)[None, :]
	gamma_pred = jnp.where(padding_mask, gamma_pred, jnp.eye(len(gamma_pred)))
	gamma_crossed = jnp.where(~jnp.isnan(padded_outputs_pred)[:, None], gamma_crossed, 0.)

	gamma_pred_U = cho_factor(gamma_pred)
	z = triangular_solve(gamma_pred_U, gamma_crossed.T).T
	y = triangular_solve(gamma_pred_U, jnp.nan_to_num(padded_outputs_pred) - post_mean_pred)

	pred_mean = post_mean_grid + (z.T @ y)
	pred_cov = gamma_grid - (z.T @ z)

	pred_cov = pred_cov #+ task_kernel.right_kernel(grid)  # Add noise after

	return pred_mean, pred_cov

@jit
def predict_single_cluster(padded_inputs_pred, padded_outputs_pred, grid,
                           post_mean_pred, post_cov_pred, post_mean_crossed, post_mean_grid, post_cov_grid,
                           pred_task_kernel) -> Tuple[Array, Array]:
		# post_mean_grid, post_cov_grid, padded_outputs_pred, mappings_pred_on_grid, grid, pred_task_kernel: BatchKernel):
	# In multi-output, we want to flatten the outputs.
	# The user should provide a specific Kernel to compute a cross-covariance with the right shape too
	padded_outputs_pred = padded_outputs_pred.reshape(padded_outputs_pred.shape[0], -1)

	return vmap(predict_single_task, in_axes=(0, 0, None, 0, 0, 0, None, None, pred_task_kernel.batch_in_axes))(
		padded_inputs_pred, padded_outputs_pred, grid,
		post_mean_pred, post_cov_pred, post_mean_crossed, post_mean_grid, post_cov_grid, pred_task_kernel.inner_kernel)
