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


def optimise_hyperparameters(mean_kernel, task_kernel, inputs, outputs, all_inputs, prior_mean, post_mean, post_cov,
                             masks, nugget=jnp.array(1e-10), max_iter=100, tol=1e-3, verbose=False):
	# Optimise mean kernel
	if verbose:
		mean_opt = optax.chain(print_info(), optax.lbfgs())
	else:
		mean_opt = optax.lbfgs()

	def mean_fun_wrapper(kern):
		res = magma_neg_likelihood(kern, all_inputs, post_mean, prior_mean, post_cov, mask=None, nugget=nugget)
		return res

	new_mean_kernel, _, mean_llh = run_opt(mean_kernel, mean_fun_wrapper, mean_opt, max_iter=max_iter, tol=tol)

	# Optimise task kernel
	if verbose:
		task_opt = optax.chain(print_info(), optax.lbfgs())
	else:
		task_opt = optax.lbfgs()

	def task_fun_wrapper(kern):
		res = magma_neg_likelihood(kern, inputs, outputs, post_mean, post_cov, mask=masks, nugget=nugget).sum()
		return res

	new_task_kernel, _, task_llh = run_opt(task_kernel, task_fun_wrapper, task_opt, max_iter=max_iter, tol=tol)

	return new_mean_kernel, new_task_kernel, mean_llh, task_llh
