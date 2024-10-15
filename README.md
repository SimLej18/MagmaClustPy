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

* The library is written in Python instead of R (obviously)
* The library is not a package, but a Python module
* We let the user chose a specific linear algebra backend. Current choices are `numpy`, `jax`, `torch` and `mlx`.
* We use matplotlib for plotting instead of ggplot2