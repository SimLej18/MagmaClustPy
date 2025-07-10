#%% md
# # Pipeline debug notebook
# 
# This notebook allows for tests and debugging inside the whole codebase of MagmaClutPy.
# 
# It removes optimisations like jax.jit and computes on 32-bit floats to allow for easier debugging and testing.
# 
# ---
#%% md
# ## Setup
#%%
USE_JIT = False
USE_X64 = False
DEBUG_NANS = False
#%%
# Standard imports
import os
if USE_X64:
	os.environ['JAX_ENABLE_X64'] = "True"

import time
from typing import NamedTuple

# JAX imports
import jax
jax.config.update("jax_disable_jit", not USE_JIT)
jax.config.update("jax_debug_nans", DEBUG_NANS)
from jax import vmap, jit
from jax.lax import cond
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.linalg import cho_solve, cho_factor
from jax.scipy.optimize import minimize
from jax.scipy.stats.multivariate_normal import logpdf
from jax.tree_util import register_pytree_node_class, tree_flatten
import chex
import optax
import optax.tree_utils as otu

# Pandas import
import pandas as pd
import numpy as np
#%% md
# ---
#%% md
# ## Preprocessing
#%%
# from MagmaClustPy.utils import preprocess_db
#%%
def preprocess_db(db: pd.DataFrame):
	"""

	:param db: the db to process, with columns "ID", "Input" and "Output"
	:return: a tuple of (all_inputs, padded_inputs, padded_outputs, masks)
	   - all_inputs: a matrix of shape (P, ) with all distinct inputs
	   - padded_inputs: a matrix of shape (M, P) where M is the number of sequences and P is the number of distinct
	   inputs. Missing inputs for each sequence are represented as NaNs.
	   - padded_outputs: a matrix of shape (M, P) with corresponding output for each input and NaNs for missing inputs
	   - masks: a matrix of shape (M, P) with 1 where the input is valid and 0 where it is padded
	"""
	# Get all distinct inputs
	all_ids = jnp.array(db["ID"].unique())
	all_inputs = jnp.sort(jnp.array(db["Input"].unique()))

	# Initialise padded inputs, padded outputs and masks
	padded_inputs = jnp.full((len(all_ids), len(all_inputs)), jnp.nan)
	padded_outputs = jnp.full((len(all_ids), len(all_inputs)), jnp.nan)
	masks = jnp.zeros((len(all_ids), len(all_inputs)), dtype=bool)

	# Fill padded inputs, padded outputs and masks
	prev_id = ""
	id_index = -1

	for row, _id, input, output in db[["ID", "Input", "Output"]].itertuples():
		if _id != prev_id:
			prev_id = _id
			id_index += 1

		idx = jnp.searchsorted(all_inputs, input)
		padded_inputs = padded_inputs.at[id_index, idx].set(input)
		padded_outputs = padded_outputs.at[id_index, idx].set(output)
		masks = masks.at[id_index, idx].set(True)

	return all_inputs, padded_inputs, padded_outputs, masks
#%% md
# ---
#%% md
# ## Kernels
#%%
# from MagmaClustPy.kernels import SEMagmaKernel, NoisySEMagmaKernel
#%%
@register_pytree_node_class
class AbstractKernel:
	def __init__(self, skip_check=False, **kwargs):
		if not skip_check:
			# Check that hyperparameters are all jnp arrays/scalars or kernels
			for key, value in kwargs.items():
				if not isinstance(value, jnp.ndarray):  # Check type
					kwargs[key] = jnp.array(value)
				if len(kwargs[key].shape) > 1:  # Check dimensionality
					raise ValueError(f"Parameter {key} must be a scalar or a 1D array, got shape {value.shape}.")

		# Register hyperparameters in *kwargs* as instance attributes
		self.__dict__.update(kwargs)

	def __str__(self):
		return f"{self.__class__.__name__}({', '.join([f'{key}={value}' for key, value in self.__dict__.items()])})"

	def __repr__(self):
		return str(self)

	@jit
	def check_kwargs(self, **kwargs):
		for key in self.__dict__:
			if key not in kwargs:
				kwargs[key] = self.__dict__[key]
		return kwargs

	@jit
	def __call__(self, x1, x2=None, **kwargs):
		# If no x2 is provided, we compute the covariance between x1 and itself
		if x2 is None:
			x2 = x1

		# Check kwargs
		kwargs = self.check_kwargs(**kwargs)

		# Call the appropriate method
		if jnp.isscalar(x1) and jnp.isscalar(x2):
			return self.compute_scalar_if_not_nan(x1, x2, **kwargs)
		elif jnp.ndim(x1) == 1 and jnp.isscalar(x2):
			return self.compute_vector_if_not_nan(x1, x2, **kwargs)
		elif jnp.isscalar(x1) and jnp.ndim(x2) == 1:
			return self.compute_vector_if_not_nan(x2, x1, **kwargs)
		elif jnp.ndim(x1) == 1 and jnp.ndim(x2) == 1:
			return self.compute_matrix(x1, x2, **kwargs)
		elif jnp.ndim(x1) == 2 and jnp.ndim(x2) == 2:
			return self.compute_batch(x1, x2, **kwargs)
		else:
			return jnp.nan

	# Methods to use Kernel as a PyTree
	def tree_flatten(self):
		return tuple(self.__dict__.values()), None  # No static values

	@classmethod
	def tree_unflatten(cls, _, children):
		# This class being abstract, this function fails when called on an "abstract instance",
		# as we don't know the number of parameters the constructor expects, yet we send it children.
		# On a subclass, this will work as expected as long as the constructor has a clear number of
		# kwargs as parameters.
		return cls(*children, skip_check=True)

	@jit
	def compute_scalar_if_not_nan(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Returns NaN if either x1 or x2 is NaN, otherwise calls the compute_scalar method.

		:param x1: scalar array
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: scalar array
		"""
		return cond(jnp.isnan(x1) | jnp.isnan(x2), lambda _: jnp.nan,
		                lambda _: self.compute_scalar(x1, x2, **kwargs), None)

	@jit
	def compute_scalar(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between two scalar arrays.

		:param x1: scalar array
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: scalar array
		"""
		return jnp.array(jnp.nan)  # To be overwritten in subclasses

	@jit
	def compute_vector(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the kernel covariance value between a vector and a scalar.

		:param x1: vector array (N, )
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: vector array (N, )
		"""
		return vmap(lambda x: self.compute_scalar_if_not_nan(x, x2, **kwargs), in_axes=0)(x1)

	@jit
	def compute_vector_if_not_nan(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Returns an array of NaN if scalar is NaN, otherwise calls the compute_vector method.

		:param x1: vector array (N, )
		:param x2: scalar array
		:param kwargs: hyperparameters of the kernel
		:return: vector array (N, )
		"""
		return cond(jnp.any(jnp.isnan(x2)), lambda _: x1 * jnp.nan, lambda _: self.compute_vector(x1, x2, **kwargs),
		                None)

	@jit
	def compute_matrix(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the kernel covariance matrix between two vector arrays.

		:param x1: vector array (N, )
		:param x2: vector array (M, )
		:param kwargs: hyperparameters of the kernel
		:return: matrix array (N, M)
		"""
		return vmap(lambda x: self.compute_vector_if_not_nan(x2, x, **kwargs), in_axes=0)(x1)

	@jit
	def compute_batch(self, x1: jnp.ndarray, x2: jnp.ndarray, **kwargs) -> jnp.ndarray:
		"""
		Compute the kernel covariance matrix between two batched vector arrays.

		:param x1: vector array (B, N)
		:param x2: vector array (B, M)
		:param kwargs: hyperparameters of the kernel. Each HP that is a scalar will be common to the whole batch, and
		each HP that is a vector will be distinct and thus must have shape (B, )
		:return: tensor array (B, N, M)
		"""
		# vmap(self.compute_matrix)(x1, x2, **kwargs)
		common_hps = {key: value for key, value in kwargs.items() if jnp.isscalar(value)}
		distinct_hps = {key: value for key, value in kwargs.items() if not jnp.isscalar(value)}

		return vmap(lambda x, y, hps: self.compute_matrix(x, y, **hps, **common_hps), in_axes=(0, 0, 0))(x1, x2,                                                                                           distinct_hps)
#%%
@register_pytree_node_class
class SEMagmaKernel(AbstractKernel):
	def __init__(self, length_scale=None, variance=None, **kwargs):
		if length_scale is None:
			length_scale = jnp.array([1.])
		if variance is None:
			variance = jnp.array([1.])
		super().__init__(length_scale=length_scale, variance=variance, **kwargs)

	@jit
	def compute_scalar(self, x1: jnp.ndarray, x2: jnp.ndarray, length_scale=None, variance=None) -> jnp.ndarray:
		return jnp.exp(variance - jnp.exp(-length_scale) * jnp.sum((x1 - x2) ** 2) * 0.5)
#%%
@register_pytree_node_class
class NoisySEMagmaKernel(AbstractKernel):
	def __init__(self, length_scale=None, variance=None, noise=None, **kwargs):
		if noise is None:
			noise = jnp.array([-1.])
		super().__init__(length_scale=length_scale, variance=variance, noise=noise, **kwargs)

	@jit
	def compute_scalar(self, x1: jnp.ndarray, x2: jnp.ndarray, length_scale=None, variance=None, noise=None) -> jnp.ndarray:
		return cond(x1 == x2,
		            lambda _: jnp.exp(variance - jnp.exp(-length_scale) * jnp.sum((x1 - x2) ** 2) * 0.5) + jnp.exp(noise),
		            lambda _: jnp.exp(variance - jnp.exp(-length_scale) * jnp.sum((x1 - x2) ** 2) * 0.5)
		            , None)
#%% md
# ---
#%% md
# ## Hyperpost
#%%
# from MagmaClustPy.hyperpost import hyperpost
#%%
@jit
def hyperpost_common_input_common_hp(outputs, prior_mean, mean_cov_u, mean_cov_inv, task_cov, inputs_to_grid=None,
                                     jitter=jnp.array(1e-10)):
	eye = jnp.eye(task_cov.shape[-1])

	# Compute task covariance and its Cholesky factor
	task_cov_u, _ = cho_factor(task_cov + eye * jitter)
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
#%%
@jit
def hyperpost_common_input_distinct_hp(outputs, prior_mean, mean_cov_u, mean_cov_inv, task_covs, inputs_to_grid=None,
                                       jitter=jnp.array(1e-10)):
	eye = jnp.broadcast_to(jnp.eye(task_covs.shape[-1]), task_covs.shape)

	# Compute task covariance and its Cholesky factor
	# task_covs_L = vmap(lambda x: cho_factor(x + eye * jitter, lower=True)[0])(task_covs)
	task_covs_u, _ = cho_factor(task_covs + eye * jitter)
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
#%%
@jit
def hyperpost_distinct_input(outputs, masks, prior_mean, mean_cov_u, mean_cov_inv, task_covs, inputs_to_grid=None,
                             jitter=jnp.array(1e-10)):
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
	# task_covs_L = vmap(lambda x: cho_factor(x + small_eye * jitter)[0])(task_covs)
	task_covs_U, _ = cho_factor(task_covs + small_eye * jitter)
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
#%%
def hyperpost(inputs, outputs, masks, prior_mean, mean_kernel, task_kernel, all_inputs=None, grid=None,
              jitter=jnp.array(1e-10)):
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
	:param jitter: jitter term to ensure numerical stability. Default is 1e-10
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
	mean_cov_u, _ = cho_factor(mean_cov + eye * jitter)
	mean_cov_inv = cho_solve((mean_cov_u, False), eye)

	if common_input:
		if common_hp:
			task_cov = task_kernel(grid)  # Shape: (N, N)
			res = hyperpost_common_input_common_hp(outputs, prior_mean, mean_cov_u, mean_cov_inv, task_cov, inputs_to_grid, jitter)
			return res

		else:  # distinct HPs, we have to compute every task covariance but no padding is required
			task_covs = task_kernel(inputs)  # Shape: (M, N, N)
			res = hyperpost_common_input_distinct_hp(outputs, prior_mean, mean_cov_u, mean_cov_inv, task_covs, inputs_to_grid, jitter)
			return res

	else:  # No common input: we have to pad and mask
		# task_covs = task_kernel(jnp.broadcast_to(all_inputs, (len(inputs), len(all_inputs))))
		task_covs = task_kernel(inputs)
		res = hyperpost_distinct_input(outputs, masks, prior_mean, mean_cov_u, mean_cov_inv, task_covs, inputs_to_grid, jitter)
		return res
#%% md
# ---
#%% md
# ## Likelihoods
#%%
from MagmaClustPy.likelihoods import magma_neg_likelihood
#%%
@jit
def solve_right_cholesky(A, B, jitter=jnp.array(1e-10)):
	""" Solves for X in X @ A = B """
	# For X @ A = B, we can transpose both sides: A.T @ X.T = B.T
	# As A and B are symmetric, this simplifies to A @ X.T = B
	# Then solve for X.T and transpose the result
	jitter_matrix = jnp.eye(A.shape[0]) * jitter
	return cho_solve(cho_factor(A + jitter_matrix), B).T
#%%
@jit
def magma_neg_likelihood_on_cov(covar, outputs, mean, mean_process_cov, mask=None, jitter=jnp.array(1e-10)):
	jitter_matrix = jnp.eye(outputs.shape[0]) * jitter

	if mask is not None:
		# Mask the covariance matrix and outputs
		mask_2D = mask[:, None] & mask[None, :]
		covar = jnp.where(mask_2D, covar, jnp.eye(outputs.shape[0]))
		outputs = jnp.where(mask, outputs, 0)
		mean = jnp.where(mask, mean, 0)
		mean_process_cov = jnp.where(mask_2D, mean_process_cov, jnp.eye(outputs.shape[0]))


	# Compute log-likelihood
	multiv_neg_log_lik = -logpdf(outputs, mean, covar + jitter_matrix)

	# Compute correction term
	correction = 0.5 * jnp.trace(solve_right_cholesky(covar, mean_process_cov, jitter=jitter))

	if mask is not None:
		# Correct log-likelihood for padding
		# The logpdf is computed as:
		# -0.5 * (N * log(2 * pi) + log(det(cov)) + (outputs - mean).T @ inv(cov) @ (outputs - mean))
		# det(cov) and the Mahalanobis distance are not affected by our padding
		# We only have to correct for the -0.5 * N * log(2 * pi) term, as N is bigger with padding
		nll_pad_correction = 0.5 * jnp.log(2 * jnp.pi) * jnp.sum(~mask, axis=0)

		# We also need to correct the correction term, as padding adds 1s to the diagonal and hence 1 to the trace
		corr_pad_correction = 0.5 * jnp.sum(~mask, axis=0)
	else:
		nll_pad_correction = 0
		corr_pad_correction = 0

	res = (multiv_neg_log_lik - nll_pad_correction) + (correction - corr_pad_correction)
	return res
#%%
@jit
def magma_neg_likelihood(kernel, inputs, outputs: jnp.array, mean: jnp.array, mean_process_cov: jnp.array, mask=None,
                         jitter=jnp.array(1e-10)):
	"""
	Computes the MAGMA log-likelihood.

	:param kernel: the kernel containing HPs to optimise. This kernel is used to compute the covariance (matrix `S`)
	:param inputs: inputs on which to compute the covariance matrix (shape (N, ))
	:param mask: boolean masks indicating which inputs and outputs to consider (shape (N, ))
	:param outputs: the observed values (shape (N, ))
	:param mean: the mean over the inputs (scalar or vector of shape (N, ))
	:param mean_process_cov: the hypper-posterior mean process covariance (matrix K^t)
	:param jitter: the jitter, for numerical stability

	:return: the negative log-likelihood (scalar)
	"""
	covar = kernel(inputs)

	# check if we need to vmap
	if inputs.ndim == 1:
		return magma_neg_likelihood_on_cov(covar, outputs, mean, mean_process_cov, mask, jitter)
	elif inputs.ndim == 2:
		return vmap(magma_neg_likelihood_on_cov, in_axes=(0, 0, None, None, 0, None))(covar, outputs, mean,
		                                                                              mean_process_cov, mask, jitter)
	else:
		raise ValueError("inputs must be either 1D or 2D")
#%% md
# ---
#%% md
# ## Hyper-parameters optimisation
#%%
# from MagmaClustPy.hp_optimisation import optimise_hyperparameters
#%%
class InfoState(NamedTuple):
	iter_num: chex.Numeric


def print_info():
	def init_fn(params):
		del params
		return InfoState(iter_num=0)

	def update_fn(updates, state, params, *, value, grad, **extra_args):
		del params, extra_args

		jax.debug.print(
			'Iteration: {i}, Value: {v}, Gradient norm: {e}',
			i=state.iter_num,
			v=value,
			e=otu.tree_l2_norm(grad),
		)
		return updates, InfoState(iter_num=state.iter_num + 1)

	return optax.GradientTransformationExtraArgs(init_fn, update_fn)
#%%
def run_opt(init_params, fun, opt, max_iter, tol):
	value_and_grad_fun = optax.value_and_grad_from_state(fun)

	def step(carry):
		params, state, prev_llh = carry
		value, grad = value_and_grad_fun(params, state=state)
		updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=fun)
		params = optax.apply_updates(params, updates)
		return params, state, value

	def continuing_criterion(carry):
		# tol is not computed on the gradients but on the difference between current and previous likelihoods, to
		# prevent overfitting on ill-defined likelihood functions where variance can blow up.
		_, state, prev_llh = carry
		iter_num = otu.tree_get(state, 'count')
		val = otu.tree_get(state, 'value')
		diff = jnp.abs(val - prev_llh)
		return (iter_num == 0) | ((iter_num < max_iter) & (diff >= tol))

	init_carry = (init_params, opt.init(init_params),
	              jnp.array(jnp.inf))  # kernel params, initial state, first iter, previous likelihood
	final_params, final_state, final_llh = jax.lax.while_loop(
		continuing_criterion, step, init_carry
	)
	return final_params, final_state, final_llh
#%%
def optimise_hyperparameters(mean_kernel, task_kernel, inputs, outputs, all_inputs, prior_mean, post_mean, post_cov,
                             masks, jitter=jnp.array(1e-10), max_iter=100, tol=1e-3, verbose=False):
	# Optimise mean kernel
	if verbose:
		mean_opt = optax.chain(print_info(),
                               optax.lbfgs(
                                   #scale_init_precond=False,
                                   linesearch=optax.scale_by_zoom_linesearch(
                                       max_linesearch_steps=50,
                                       verbose=True,
                                       initial_guess_strategy='one'
                                   )
                               )
                               )
	else:
		mean_opt = optax.lbfgs()

	def mean_fun_wrapper(kern):
		res = magma_neg_likelihood(kern, all_inputs, post_mean, prior_mean, post_cov, mask=None, jitter=jitter)
		return res

	new_mean_kernel, _, mean_llh = run_opt(mean_kernel, mean_fun_wrapper, mean_opt, max_iter=max_iter, tol=tol)

	# Optimise task kernel
	if verbose:
		task_opt = optax.chain(print_info(),
                               optax.lbfgs(
                                   #scale_init_precond=False,
                                   linesearch=optax.scale_by_zoom_linesearch(
                                       max_linesearch_steps=50,
                                       verbose=True,
                                       initial_guess_strategy='one'
                                   )
                               )
                               )
	else:
		task_opt = optax.lbfgs()

	def task_fun_wrapper(kern):
		res = magma_neg_likelihood(kern, inputs, outputs, post_mean, post_cov, mask=masks, jitter=jitter).sum()
		return res

	new_task_kernel, _, task_llh = run_opt(task_kernel, task_fun_wrapper, task_opt, max_iter=max_iter, tol=tol)

	return new_mean_kernel, new_task_kernel, mean_llh, task_llh
#%% md
# ---
#%% md
# ## Run experiment
#%% md
# ### Config
#%%
MAX_ITER = 25
CONVERG_THRESHOLD = 1e-10
jitter = jnp.array(1e-5)
verbose = False
#%%
dataset = "small"
grids = {
	"small": jnp.arange(-10, 10, 0.5),
	"medium": jnp.arange(-100, 100, 0.5),
	"large": jnp.arange(-500, 500, 0.5),
	"custom": jnp.arange(-20, 20, 0.5)
}
grid = grids[dataset] if dataset in grids else grids["custom"]
common_input = False
common_hp = True
#%% md
# ### Start timer
#%%
start = time.time()
#%% md
# ### Data import
#%%
db = pd.read_csv(f"../dummy_datasets/{dataset}_{'common_input' if common_input else 'distinct_input'}_{'common_hp' if common_hp else 'distinct_hp'}.csv")
# db has 3 columns: ID, Input, Output
#%%
# First 90% of IDs are for training, last 10% for testing
train_ids = db["ID"].unique()  # for debug
test_ids = []  # for debug
#train_ids = db["ID"].unique()[:int(0.9 * db["ID"].nunique())]
#test_ids = db["ID"].unique()[int(0.9 * db["ID"].nunique()):]

db_train = db[db["ID"].isin(train_ids)]
db_test = db[db["ID"].isin(test_ids)]

# N.b: data is already sort by ID and Input in the toy datasets, but in a real case scenario, we would need to sort it
#%% md
# ### Data preprocessing
#%%
# We need to convert the dataframe into jax arrays
# inputs: (M, N) timestamps
# outputs: (M, N) observed outputs
# unique_inputs: (P,) unique timestamps (if common_input, P = N)
all_inputs_train, padded_inputs_train, padded_outputs_train, masks_train = preprocess_db(db_train)
all_inputs_train.shape, padded_outputs_train.shape
#%%
np.asarray(padded_outputs_train)
#%%
np.asarray(masks_train)
#%% md
# ### Training
#%%
# Priors
prior_mean = jnp.zeros_like(all_inputs_train)
mean_kernel = SEMagmaKernel(length_scale=0.9, variance=1.5)

if common_hp:
	task_kernel = NoisySEMagmaKernel(length_scale=0.3, variance=1., noise=-2.5)
else:
	task_kernel = NoisySEMagmaKernel(length_scale=jnp.array([0.3] * padded_inputs_train.shape[0]), variance=jnp.array([1.] * padded_inputs_train.shape[0]), noise=jnp.array([-2.5] * padded_inputs_train.shape[0]))
#%%
prev_mean_llh = jnp.inf
prev_task_llh = jnp.inf
conv_ratio = jnp.inf

for i in range(MAX_ITER):
	print(f"Iteration {i:4}\tLlhs: {prev_mean_llh:12.4f}, {prev_task_llh:12.4f}\tConv. Ratio: {conv_ratio:.5f}\t\n\tMean: {mean_kernel}\t\n\tTask: {task_kernel}")
	# e-step: compute hyper-posterior
	post_mean, post_cov = hyperpost(padded_inputs_train, padded_outputs_train, masks_train, prior_mean, mean_kernel, task_kernel, all_inputs=all_inputs_train, jitter=jitter)

	# m-step: update hyperparameters
	mean_kernel, task_kernel, mean_llh, task_llh = optimise_hyperparameters(mean_kernel, task_kernel, padded_inputs_train, padded_outputs_train, all_inputs_train, prior_mean, post_mean, post_cov, masks_train, jitter=jitter, verbose=verbose)

	# Check convergence
	if i > 0:
		conv_ratio = jnp.abs((prev_mean_llh + prev_task_llh) - (mean_llh + task_llh)) / jnp.abs(prev_mean_llh + prev_task_llh)
		if conv_ratio < CONVERG_THRESHOLD:
			print(f"Convergence reached after {i+1} iterations.\tLlhs: {mean_llh:12.4f}, {task_llh:12.4f}\n\tMean: {mean_kernel}\n\tTask: {task_kernel}")
			break

	if i == MAX_ITER - 1:
		print(f"WARNING: Maximum number of iterations reached.\nLast modif: {jnp.abs(prev_mean_llh - mean_llh).item()} & {jnp.abs(prev_task_llh - task_llh).item()}")

	prev_mean_llh = mean_llh
	prev_task_llh = task_llh
#%% md
# ### Prediction
#%%

#%%

#%% md
# ### End timer
#%%
end = time.time()
#%%
print(f"Magma finished in {end - start:.2f}s")
#%% md
# ---
#%% md
# ## Post-experiment sandbox
#%%
mean_kernel
#%%
task_kernel
#%% md
# ---