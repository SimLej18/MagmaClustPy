from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu

from MagmaClustPy.likelihoods import magma_neg_likelihood


# Taken from optax doc (https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html#l-bfgs-solver)
class InfoState(NamedTuple):
	iter_num: chex.Numeric


def print_info():
	"""
	:return: Basic optax transformation that prints the iteration number, value, and gradient norm at each step.
	"""
	def init_fn(params):
		del params
		return InfoState(iter_num=0)

	def update_fn(updates, state, params, *, value, grad, **extra_args):
		del params, extra_args

		jax.debug.print(
			'Iteration: {i}, Value: {v}, Gradient norm: {e}',
			i=state.iter_num,
			v=value,
			e=otu.tree_norm(grad),
		)
		return updates, InfoState(iter_num=state.iter_num + 1)

	return optax.GradientTransformationExtraArgs(init_fn, update_fn)


# Adapted from optax doc (https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html#l-bfgs-solver)
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


def optimise_hyperparameters(mean_kernel, task_kernel, inputs, outputs, mappings, all_inputs, prior_mean, post_mean,
                             post_cov, jitter=jnp.array(1e-10), max_iter=100, tol=1e-3, verbose=False):
	"""
	Optimise the hyperparameters of the mean and task kernels using L-BFGS and the corrected likelihood of Magma.

	:param mean_kernel: Kernel to optimise the mean process covariance.
	:param task_kernel: Kernel to optimise the task covariance.
	:param inputs: Inputs of every point, for every task, padded with NaNs. Shape (T, Max_N_i, I)
	:param outputs: Outputs of every point, for every task, padded with NaNs. Shape (T, Max_N_i, O)
	:param mappings: Indices of every input in the all_inputs array, padded with len(all_inputs). Shape (T, Max_N_i)
	:param all_inputs: all distinct inputs. Shape (N, I)
	:param prior_mean: prior mean over all_inputs or grid if provided. Shape (N,) or scalar if constant
	across the domain.
	:param post_mean: hyperpost mean over all_inputs. Shape (N,)
	:param post_cov: hyperpost covariance over all_inputs. Shape (N, N)
	:param jitter: jitter term to ensure numerical stability. Default is 1e-10
	:param max_iter: maximum number of iterations for the optimisation, default is 100.
	:param tol: the optimisation stops when the change in likelihood is below this threshold, default is 1e-3.
	:param verbose: if True, prints the optimisation progress, default is False.

	:return: A tuple of the optimised mean kernel, task kernel, mean log-likelihood, and task log-likelihood.
	"""

	# Optimise mean kernel
	if verbose:
		mean_opt = optax.chain(print_info(), optax.lbfgs())
	else:
		mean_opt = optax.lbfgs()

	def mean_fun_wrapper(kern):
		res = magma_neg_likelihood(kern, all_inputs, post_mean, None, prior_mean, post_cov, jitter=jitter)
		return res

	new_mean_kernel, _, mean_llh = run_opt(mean_kernel, mean_fun_wrapper, mean_opt, max_iter=max_iter, tol=tol)

	# Optimise task kernel
	if verbose:
		task_opt = optax.chain(print_info(), optax.lbfgs())
	else:
		task_opt = optax.lbfgs()

	def task_fun_wrapper(kern):
		res = magma_neg_likelihood(kern, inputs, outputs, mappings, post_mean, post_cov, jitter=jitter).sum()
		return res

	new_task_kernel, _, task_llh = run_opt(task_kernel, task_fun_wrapper, task_opt, max_iter=max_iter, tol=tol)

	return new_mean_kernel, new_task_kernel, mean_llh, task_llh
