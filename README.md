# MagmaClustPy
---

MagmaClustPy is a probabilistic learning framework based on MagmaClust, a multi-task Gaussian Process framework.

An original implementation of MagmaClust is available as [a R package](https://github.com/ArthurLeroy/MagmaClustR).

This implementation has many limitations:
* it doesn't do parallel computations and doesn't run on GPU
* it doesn't support non-gaussian likelihoods (for classification for example)
* it only models single-output GPs
* it trains on all the data at once, scaling pretty badly on bigger datasets

This Python package will aleviate these limitations with multiple design choices:
* We use Jax along with mapping/padding methods to allow fast and parallel computations on CPU/GPU/TPU
* We develop a multi-output algorithm based on Process Convolution for joint optimisation of correlated outputs
* We use Laplace-Matching to adapt to non-gaussian data and problems
* We support data points coming with known, heteroskedastic uncertainty estimates
* We learn data in batches to reduce time of training
* We explore sparse covariance matrix computations to speed up inference

---

## Installation

You can install a minimal version of the library using:

```bash
pip install magma-py-minimal
```

NOTE: for now, this minimal version is only compatible with python >= 3.12.

NOTE: this version is a prototype and comes with no guarantee of stability. 
For now, it is not advised to use it for production/scientific contributions. 
Stable version is planned for summer 2026.

## Main differences with the original MagmaClustR library

* This is a module written in Python instead of package coded in R (obviously)
* The package runs on JAX and can therefore leverage various backends (CPU, GPU, TPU). 
* We use **custom classes for kernels** rather than string identifiers. These kernels can be composed (à la GPytorch). 
You can find them in `kernels.py`. Therefore, *signatures of functions that use kernels might be different*. A common 
example of this is the initialisation of kernel HPs. Rather than sending the kernel class and HPs as separate arguments, 
**the user can initialise the kernel with the wanted HP and then send it as a single argument**.
* We use **matplotlib** for plotting instead of **ggplot2**
* Files, class names, functions names, variables and parameters might have different names to be clearer or respect
Python conventions.
* This library sticks with the default precision of the linear algebra backend (or the one specified by the user). No 
implicit rounding of numbers is performed by the library itself.
* This library uses `logging` instead of `cat`. You can configure the logging level like this: 
`logging.basicConfig(level=logging.INFO)`.

---

## Development roadmap

- [x] Cluster mixture init and update
- [x] Cluster hyperpost and HP optim
- [ ] Model classes
- [x] Cluster prediction
- [x] Plot utilities
- [ ] Initializers
- [ ] Prior means modules
- [ ] Likelihood modules
- [ ] Minimal documentation (guides and API)
- [ ] PyPI package and deployment setup

🚀 Alpha release !

- [ ] Bug test - issue management
- [ ] Unit test
- [ ] Multi-output GPs
- [ ] Complete documentation
- [ ] Contribution guides
- [ ] Dev pipeline tools for testing/coverage/...

🚀 1.0.0 release

- [ ] Laplace-Matching likelihoods
- [ ] Continued development

---

## Help, feedback, contributions

Any feedback, issue or contribution is obviously mor than welcome! 
Don't hesitate to open an issue/discussion on GitHub, or get in touch with [Arthur Leroy](https://arthur-leroy.netlify.app/) if you have any question.

