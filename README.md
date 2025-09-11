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

## Status of the implementation

This package is currently **a work in progress** and is not yet functional. 
Currently, it's used to test our new design choices.
The training and prediction pipeline of a Magma model is mostly functional however. 
Benchmarks demonstrate an impressive speed-up over the R version on datasets containing 600 tasks, each with 450 points on a 2000 points grid.

Most of the features will be implemented in the following months, likely in this order:
* clustering
* heteroskedasticity
* multi-output
* non-gaussian likelihood
* batch learning
* sparse precision matrices approximations

---

## Installation

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

## Help, feedback, contributions

Any feedback, issue or contribution is obviously mor than welcome! 
Don't hesitate to open an issue/discussion on GitHub, or get in touch with [Arthur Leroy](https://arthur-leroy.netlify.app/) if you have any question.
