# MagmaClustPy
---

MagmaClustPy is a Python translation of the [MagmaClustR](https://github.com/ArthurLeroy/MagmaClustR) library.

For now, the translation is a work in progress and the library is not yet functional. The goal is to provide a Python 
version of the MagmaClustR library, with the same functionalities, API and results.

---

## Installation

TODO

---

## Main differences with the original MagmaClustR library

* This is a module written in Python instead of package coded in R (obviously)
* We let the user chose a **specific linear algebra backend**. The only complete backend is `numpy` for now, but we 
plan to include `jax`, `torch` and `mlx`. When a specific feature is missing in a backend, the module reverts to 
`numpy` (which may deteriorate performances).
* We use **custom classes for kernels** rather than string identifiers. These kernels can be composed (à la GPytorch). 
You can find them in `kernels.py`. Therefore, *signature of functions that use kernels might be different*. A common 
example of this is the initialisation of kernel HPs. Rather than sending the kernel class and HPs as separate arguments, 
**the user can initialise the kernel with the wanted HP and then send it as a single argument**.
* We use **matplotlib** for plotting instead of **ggplot2**
* Files, class names, functions names, variables and parameters follow the **Python naming conventions**. Some variables 
used inside functions might have different names when that makes the code more understandable.
* This library sticks with the default precision of the linear algebra backend (or the one specified by the user). No 
implicit rounding of numbers is performed by the library itself.
* This library uses `logging` instead of `cat`. You can configure the logging level like this: 
`logging.basicConfig(level=logging.INFO)`.