from typing import Tuple, Callable, Any, Optional
from functools import partial

import jax
from jax import jit, vmap, Array
import jax.numpy as jnp
import optax
import optax.tree_utils as otu

from kernax import AbstractKernel
from MagmaClustPy.likelihoods import magma_nll


# Adapted from optax doc (https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html#l-bfgs-solver)
@partial(jit, static_argnames=('likelihood', 'opt', 'max_iter', 'tol'))
def optimise_kernel(init_kernel: AbstractKernel, likelihood: Callable,
                    opt: optax.GradientTransformationExtraArgs, max_iter: int, tol: float) -> Tuple[AbstractKernel, Any, Array]:
	# L-BFGS with linesearch requires a custom value_and_grad function.
	likelihood_value_and_grad = optax.value_and_grad_from_state(likelihood)

	@jit
	def step(carry):
		kernel, state, prev_llh = carry
		value, grad = likelihood_value_and_grad(kernel, state=state)
		updates, state = opt.update(grad, state, kernel, value=value, grad=grad, value_fn=likelihood)
		kernel = optax.apply_updates(kernel, updates)
		return kernel, state, value

	@jit
	def continuing_criterion(carry):
		# tol is not computed on the gradients but on the difference between current and previous likelihoods, to
		# prevent overfitting on ill-defined likelihood functions where variance can blow up.
		_, state, prev_llh = carry
		iter_num = otu.tree_get(state, 'count')
		val = otu.tree_get(state, 'value')
		diff = jnp.abs(val - prev_llh)
		return (iter_num == 0) | ((iter_num < max_iter) & (diff >= tol))

	# kernel params, initial state, first iter, previous likelihood
	init_carry = (init_kernel, opt.init(init_kernel), jnp.array(jnp.inf))
	final_kernel, final_state, final_llh = jax.lax.while_loop(
		continuing_criterion, step, init_carry
	)
	return final_kernel, final_state, final_llh


@partial(jit, static_argnames=('max_iter', 'tol'))
def optimise_mean_kernel(mean_kernel: AbstractKernel, all_inputs: Array, prior_mean: Array,
						 post_means: Array, post_covs: Array,
						 jitter: Array = jnp.array(1e-5),
						 max_iter: int = 100, tol: float = 1e-3) -> Tuple[AbstractKernel, Array]:
	"""
	Optimise the hyperparameters of the mean kernel using L-BFGS and the corrected likelihood of Magma.

	:param mean_kernel: Kernel to optimise the mean process covariance.
	:param all_inputs: all distinct inputs. Shape (N, I)
	:param prior_mean: prior mean over all_inputs or grid if provided. Shape (N,)
	:param post_means: hyperpost mean over all_inputs. Shape (N,)
	:param post_covs: hyperpost covariance over all_inputs. Shape (N, N)
	:param jitter: jitter term to ensure numerical stability. Default is 1e-5
	:param max_iter: maximum number of iterations for the optimisation, default is 100.
	:param tol: the optimisation stops when the change in likelihood is below this threshold, default is 1e-3.

	:return: A tuple of the optimised mean kernel and mean log-likelihood.
	"""
	opt = optax.lbfgs()

	def fun_wrapper(kern):
		cluster_magma_nll = vmap(magma_nll, in_axes=(None, None, 0, None, 0, None, None))
		res = cluster_magma_nll(kern, all_inputs, post_means, prior_mean, post_covs, None, jitter)
		return res.sum()

	new_mean_kernel, _, mean_llh = optimise_kernel(mean_kernel, fun_wrapper, opt, max_iter=max_iter, tol=tol)

	return new_mean_kernel, mean_llh


@partial(jit, static_argnames=('max_iter', 'tol', 'shared_hp', 'cluster_hp'))
def optimise_task_kernel(task_kernel: AbstractKernel, inputs: Array, outputs: Array,
						 mappings: Array, post_means: Array, post_covs: Array,
                         shared_hp: bool, cluster_hp: bool, mixture_coeffs: Optional[Array] = None,
						 jitter: Array = jnp.array(1e-5), max_iter: int = 100, tol: float = 1e-3) \
		-> Tuple[AbstractKernel, Array]:
	"""Optimise the hyperparameters of the task kernel using L-BFGS and the corrected likelihood of Magma.

	:param task_kernel: Kernel to optimise the task covariance.
	:param inputs: Inputs of every point, for every task, padded with NaNs. Shape (T, Max_N_i, I)
	:param outputs: Outputs of every point, for every task, padded with NaNs. Shape (T, Max_N_i, O)
	:param mappings: Indices of every input in the all_inputs array, padded with len(all_inputs). Shape (T, Max_N_i)
	across the domain.
	:param post_means: hyperpost mean over all_inputs. Shape (N,)
	:param post_covs: hyperpost covariance over all_inputs. Shape (N, N)
	:param shared_hp: whether the kernel uses shared hyperparameters.
	:param cluster_hp: whether the kernel uses cluster hyperparameters.
	:param mixture_coeffs: coefficients of the clustering mixture. Shape (K, T)
	:param jitter: jitter term to ensure numerical stability. Default is 1e-5
	:param max_iter: maximum number of iterations for the optimisation, default is 100.
	:param tol: the optimisation stops when the change in likelihood is below this threshold, default is 1e-3.

	:return: A tuple of the optimised task kernel and mean log-likelihood.
	"""
	opt = optax.lbfgs()

	def task_fun_wrapper(kern):
		task_batched_magma_nll = vmap(magma_nll, in_axes=(None if shared_hp else 0, 0, 0, None, None, 0, None))
		cluster_batched_magma_nll = vmap(
			lambda k, *args: task_batched_magma_nll(k.inner if cluster_hp else k, *args),
			in_axes=(0 if cluster_hp else None, None, None, 0, 0, None, None))
		res = cluster_batched_magma_nll(kern.inner, inputs, outputs, post_means, post_covs, mappings, jitter)

		if mixture_coeffs is None or (not shared_hp and cluster_hp):
			# No need to ponderate by mixture coefficients, we want to optimise every parameter
			return res.sum()

		# In every other scenarios, we multiply by mixture_coeffs
		return (res * mixture_coeffs).sum()

	new_task_kernel, _, task_llh = optimise_kernel(task_kernel, task_fun_wrapper, opt, max_iter=max_iter, tol=tol)

	return new_task_kernel, task_llh
