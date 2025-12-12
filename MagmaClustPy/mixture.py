from typing import Optional
from functools import partial

from jax import jit, Array
import jax.numpy as jnp
from kernax import AbstractKernel

from MagmaClustPy.likelihoods import compute_task_nlls


@partial(jit, static_argnums=(6, 7))
def update_mixture(task_kernel: AbstractKernel, inputs: Array, outputs: Array,
					 mappings: Array, post_means: Array, post_covs: Array,
	                 shared_hp: bool, cluster_hp: bool,
					 jitter: Optional[Array] = jnp.array(1e-5)):
	task_nlls = -compute_task_nlls(task_kernel, inputs, outputs, mappings, post_means, post_covs, shared_hp, cluster_hp, jitter)

	# Make probabilities sum to 1 across tasks, by applying softmax
	new_mixture = jnp.exp(task_nlls - jnp.max(task_nlls, axis=0, keepdims=True))
	new_mixture /= jnp.sum(new_mixture, axis=0, keepdims=True)

	return new_mixture