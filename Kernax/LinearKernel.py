from jax import jit
from jax.tree_util import register_pytree_node_class
from jax import numpy as jnp

from functools import partial

from .AbstractKernel import StaticAbstractKernel, AbstractKernel


class StaticLinearKernel(StaticAbstractKernel):
    @classmethod
    @partial(jit, static_argnums=(0,))
    def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the linear kernel covariance value between two vectors.
        
        :param kern: the kernel to use, containing hyperparameters (variance, c).
        :param x1: scalar array.
        :param x2: scalar array.
        :return: scalar array (covariance value).
        """
        x1_shifted = x1 - kern.offset_c
        x2_shifted = x2 - kern.offset_c
        
        # Compute the dot product of the shifted vectors
        dot_product = jnp.sum(x1_shifted * x2_shifted)
        
        return kern.variance_b + kern.variance_v * dot_product


@register_pytree_node_class
class LinearKernel(AbstractKernel):
    def __init__(self, variance_b=None, variance_v=None, offset_c=None, **kwargs):
        """
        :param variance_b: Bias variance (σ²_b). Controls the vertical offset.
        :param variance_v: Weight variance (σ²_v). Controls the slope.
        :param offset_c: Input offset (c). Determines the crossing point of the functions.
        """
        super().__init__(
            variance_b=variance_b, 
            variance_v=variance_v, 
            offset_c=offset_c, 
            **kwargs
        )
        
        self.static_class = StaticLinearKernel