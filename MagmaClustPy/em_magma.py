from typing import Any, Dict

import pandas as pd

from MagmaClustPy import lin_alg_backend as lab
from MagmaClustPy import config
from MagmaClustPy.kernels import Kernel


def e_step(db: pd.DataFrame,
           m_0: lab.array.type,
           kern_0: Kernel,
           kern_i: Kernel,
           pen_diag: float = config["pen_diag"],
           all_inputs: lab.array.type = None) -> Dict[str, Any]:
	"""
	E-Step of the EM algorithm

	Expectation step of the EM algorithm to compute the parameters of the
	hyper-posterior Gaussian distribution of the mean process in Magma.

	:param db: A pandas DataFrame. Columns required: ID, Input, Output.
		Additional columns for covariates can be specified.
	:type db: pd.DataFrame
	:param m_0: A numpy array, corresponding to the prior mean of the mean GP.
	:type m_0: np.ndarray
	:param kern_0: A kernel function, associated with the mean GP.
	:type kern_0: Callable
	:param kern_i: A kernel function, associated with the individual GPs.
	:type kern_i: Callable
	:param pen_diag: A float. A jitter term, added on the diagonal to prevent
		numerical issues when inverting nearly singular matrices. Default is 1e-10.
	:type pen_diag: float
	:param all_inputs: A lab array, containing all the distinct Input values
		in the `db` argument.
	:type all_inputs: lab.array.type
	:return: A dictionary containing the elements 'mean', a pandas DataFrame
		containing the Input and associated Output of the hyper-posterior's mean
		parameter, and 'cov', the hyper-posterior's covariance matrix.
	:rtype: Dict[str, Any]

	:keywords: internal

	:examples: TRUE
	"""
	# Extract unique inputs
	if all_inputs is None:
		all_inputs = db['Input'].unique()

	# Compute all inverse covariance matrices
	inv_0 = kern_0.inverse_covariance_matrix(all_inputs, pen_diag=pen_diag)
	print(inv_0)


def m_step():
	pass
