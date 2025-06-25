# MagmaClustPy
---

MagmaClustPy is a Python translation of the [MagmaClustR](https://github.com/ArthurLeroy/MagmaClustR) library.

For now, the translation is a work in progress and the library is not yet functional. The goal is to provide a Python 
version of the MagmaClustR library, with the same functionalities and results. The API however might be different.

---

## Installation

TODO

To run the code in this repository, you have to setup a Python environment. You can either load the conda environment 
from env/`environment.yml` or create a new one and install the libraries using the `requirements.txt` file.

```bash
conda env create -f env/environment.yml
conda activate MagmaClustPy
```
or

```bash
python -m venv MagmaClustPy
source MagmaClustPy/bin/activate
pip install -r env/requirements.txt
```


---

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