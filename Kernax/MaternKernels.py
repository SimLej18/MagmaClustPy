from jax import jit
from jax.tree_util import register_pytree_node_class
from jax import numpy as jnp

from functools import partial

# Assuming Kernax is a module in your project that defines these base classes
from Kernax import StaticAbstractKernel, AbstractKernel


# Matern 1/2 (Exponential) Kernel defined in Rasmussen and Williams (2006), section 4.2
class StaticMatern12Kernel(StaticAbstractKernel):
    @classmethod
    @partial(jit, static_argnums=(0,))
    def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the Matern 1/2 kernel covariance value between two vectors.

        :param kern: the kernel to use, containing hyperparameters (length_scale)
        :param x1: scalar array
        :param x2: scalar array
        :return: scalar array
        """
        r = jnp.linalg.norm(x1 - x2)  # Euclidean distance
        return jnp.exp(-r / kern.length_scale)


@register_pytree_node_class
class Matern12Kernel(AbstractKernel):
    def __init__(self, length_scale=None):
        super().__init__(length_scale=length_scale)
        self.static_class = StaticMatern12Kernel


# Matern 3/2 Kernel defined in Rasmussen and Williams (2006), section 4.2
class StaticMatern32Kernel(StaticAbstractKernel):
    @classmethod
    @partial(jit, static_argnums=(0,))
    def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the Matern 3/2 kernel covariance value between two vectors.

        :param kern: the kernel to use, containing hyperparameters (length_scale)
        :param x1: scalar array
        :param x2: scalar array
        :return: scalar array
        """
        r = jnp.linalg.norm(x1 - x2)  # Euclidean distance
        sqrt3_r_div_l = (jnp.sqrt(3) * r) / kern.length_scale
        return (1.0 + sqrt3_r_div_l) * jnp.exp(-sqrt3_r_div_l)


@register_pytree_node_class
class Matern32Kernel(AbstractKernel):
    def __init__(self, length_scale=None):
        super().__init__(length_scale=length_scale)
        self.static_class = StaticMatern32Kernel


# Matern 5/2 Kernel defined in Rasmussen and Williams (2006), section 4.2
class StaticMatern52Kernel(StaticAbstractKernel):
    @classmethod
    @partial(jit, static_argnums=(0,))
    def pairwise_cov(cls, kern, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the Matern 5/2 kernel covariance value between two vectors.

        :param kern: the kernel to use, containing hyperparameters (length_scale)
        :param x1: scalar array
        :param x2: scalar array
        :return: scalar array
        """
        r = jnp.linalg.norm(x1 - x2)  # Euclidean distance
        sqrt5_r_div_l = (jnp.sqrt(5) * r) / kern.length_scale
        return (1.0 + sqrt5_r_div_l + (5.0 / 3.0) * (r / kern.length_scale)**2) * jnp.exp(-sqrt5_r_div_l)


@register_pytree_node_class
class Matern52Kernel(AbstractKernel):
    def __init__(self, length_scale=None):
        super().__init__(length_scale=length_scale)
        self.static_class = StaticMatern52Kernel