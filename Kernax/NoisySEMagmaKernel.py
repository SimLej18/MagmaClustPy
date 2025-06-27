from jax import jit
from jax.tree_util import register_pytree_node_class
from jax import numpy as jnp
from jax.lax import cond

from Kernax import SEMagmaKernel, AbstractKernel


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
