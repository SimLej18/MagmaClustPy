import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu

from MagmaClustPy.likelihoods import magma_neg_likelihood


# Taken from optax doc (https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html#l-bfgs-solver)
def run_opt(init_params, fun, opt, max_iter, tol):
	value_and_grad_fun = optax.value_and_grad_from_state(fun)

	def step(carry):
		params, state = carry
		value, grad = value_and_grad_fun(params, state=state)
		updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=fun)
		params = optax.apply_updates(params, updates)
		return params, state

	def continuing_criterion(carry):
		_, state = carry
		iter_num = otu.tree_get(state, 'count')
		grad = otu.tree_get(state, 'grad')
		err = otu.tree_l2_norm(grad)
		return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

	init_carry = (init_params, opt.init(init_params))
	final_params, final_state = jax.lax.while_loop(
		continuing_criterion, step, init_carry
	)
	return final_params, final_state


def optimise_hyperparameters(mean_kernel, task_kernel, inputs, outputs, all_inputs, prior_mean, post_mean, post_cov,
                             masks, nugget=jnp.array(1e-10)):
	big_nugget = jnp.eye(all_inputs.shape[0]) * nugget

	# Optimise mean kernel
	mean_opt = optax.lbfgs()
	mean_fun_wrapper = lambda kern: magma_neg_likelihood(kern, all_inputs, prior_mean, post_mean, post_cov, mask=None,
	                                                     nugget=nugget)

	new_mean_kernel, _ = run_opt(mean_kernel, mean_fun_wrapper, mean_opt, max_iter=100, tol=1e-3)

	# Optimise task kernel
	task_opt = optax.lbfgs()
	task_fun_wrapper = lambda kern: magma_neg_likelihood(kern, inputs, outputs, prior_mean, post_cov, mask=masks,
	                                                     nugget=nugget).sum()

	new_task_kernel, _ = run_opt(task_kernel, task_fun_wrapper, task_opt, max_iter=100, tol=1e-2)

	return new_mean_kernel, new_task_kernel
