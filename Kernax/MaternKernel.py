from jax import jit
from jax.tree_util import register_pytree_node_class
from jax import numpy as jnp
from jax.scipy.special import gamma, kv # Special functions for Matern kernel
from functools import partial

from Kernax import StaticAbstractKernel, AbstractKernel

# FIXME: Matern kernel uses Bessel functions, which are not available yet in JAX.
# See: https://github.com/jax-ml/jax/issues/11002 and GPJax repo to see how they implemented it.

class StaticMaternKernel(StaticAbstractKernel):
    @classmethod
    @partial(jit, static_argnums=(0,))
    def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
		Compute the kernel covariance value between two vectors.

		:param kern: the kernel to use, containing hyperparameters
		:param x1: scalar array
		:param x2: scalar array
		:return: scalar array
		"""
        # Compute the Euclidean distance. Note: sum((x1-x2)**2) is the squared distance.
        d = jnp.sqrt(jnp.sum((x1 - x2) ** 2))

        # Formule du noyau Matérn :
        # C(d) = σ² * (2**(1-ν) / Γ(ν)) * (√(2ν)d/l)**ν * K_ν(√(2ν)d/l)
        # où σ²=variance, l=length_scale, ν=nu.

        # On pré-calcule l'argument des fonctions pour la clarté
        arg = jnp.sqrt(2 * kern.nu) * d / kern.length_scale
        
        # Calcul des deux parties principales de la formule
        const_factor = kern.variance * (2**(1 - kern.nu) / gamma(kern.nu))
        main_term = (arg**kern.nu) * kv(kern.nu, arg)

        # WARNING : Stability for d=0
        # The case where d=0 is a known limit that equals kern.variance.
        # We use jnp.where to ensure numerical stability, because for d=0,arg=0, and the computation of `main_term` can give NaN (0 * infinity).
        return jnp.where(d > 0.0, const_factor * main_term, kern.variance)

@register_pytree_node_class
class MaternKernel(AbstractKernel):
    def __init__(self, length_scale=None, variance=None, nu=1.5, **kwargs):
        """
        Initialize the Matern kernel.

        :param length_scale: Lengthscale parameter (l).
        :param variance: Variance parameter (σ²).
        :param nu: Smoothness parameter (ν). Common values: 0.5, 1.5, 2.5.
        """
        super().__init__(length_scale=length_scale, variance=variance, **kwargs)
        self.nu = nu
        self.static_class = StaticMaternKernel