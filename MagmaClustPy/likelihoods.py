from typing import Union
import pandas as pd

from MagmaClustPy import lin_alg_backend as lab
from MagmaClustPy.kernels import Kernel

# TODO: Some of these functions do not use standard formulas for the likelihoods.
#  Ask Leroy about it and maybe provide alternatives.


def log_likelihood_gp_mod(
		db: pd.DataFrame,
		mean: lab.array.type,
		kern: Kernel,
		post_cov: lab.array.type,
		pen_diag: float
) -> float:
	"""
	Modified log-Likelihood function for Gaussian Processes.
	In MagmaClustR, this function is named `logL_GM_mod()`.

	Computes the log-Likelihood function used in Magma during the maximisation step of
	the training. The log-Likelihood is defined as a simple Gaussian likelihood
	with an additional correction trace term.

	:param db: DataFrame containing values to compute logL on.
		Required columns: 'Input', 'Output'. Additional covariate columns are allowed.
	:type db: pd.DataFrame
	:param mean: Vector specifying the mean of the GP at the reference inputs.
		If scalar, will be broadcast to match the number of observations.
	:type mean: lab.array.type
	:param kern: Kernel object with a method to compute the inverse covariance matrix.
	:type kern: Kernel
	:param post_cov: Covariance parameter of the hyper-posterior.
		Used to compute the correction term.
	:type post_cov: lab.array.type
	:param pen_diag: Small positive jitter term added to the covariance matrix to avoid
		numerical issues when inverting nearly singular matrices.
	:type pen_diag: float

	:returns: Value of the modified Gaussian log-Likelihood defined in Magma.
	:rtype: float

	:notes: The function computes two terms:
	1. Classical Gaussian log-likelihood
	2. Correction trace term (- 1/2 * Trace(inv @ post_cov))
	FIXME: Result from this function is close but not exactly the same as MagmaClustR. Investigate.
	"""
	if mean.size == 1:
		mean = lab.full(len(db), mean)

	# Extract the input variables (reference Input + Covariates)
	# TODO: covariates are not yet supported
	inputs = db['Input'].values

	# Compute the inverse of the covariance matrix
	inv_cov = kern.inverse_covariance_matrix(inputs, pen_diag)

	# Compute multivariate normal log density
	output = db['Output'].values
	diff = output - mean

	# Classical Gaussian log-likelihood (negative because we're minimizing)
	log_det = lab.linalg.slogdet(inv_cov)[1]  # get the log determinant
	quad_form = diff @ inv_cov @ diff
	n = len(output)
	LL_norm = -(-0.5 * (n * lab.log(2 * lab.pi) + log_det + quad_form))

	# Correction trace term
	cor_term = 0.5 * lab.sum(inv_cov * post_cov)

	return float(LL_norm + cor_term)