from typing import Tuple
import logging

import pandas as pd
from scipy.optimize import minimize

from MagmaClustPy import lin_alg_backend as lab
from MagmaClustPy import config
from MagmaClustPy.kernels import Kernel, SquaredExponentialMagmaKernel
from MagmaClustPy.likelihoods import log_likelihood_gp_mod


def e_step(db: pd.DataFrame,
           m_0: lab.array.type,
           kern_0: Kernel,
           kern_i: Kernel,
           pen_diag: float = config["pen_diag"],
           all_inputs: lab.array.type = None) -> Tuple[lab.array.type, lab.array.type]:
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
	:return: A tuple containing the posterior mean and covariance of the mean process.
	:rtype: Tuple[lab.array.type, lab.array.type]

	:keywords: internal

	:examples: TRUE
	"""
	# Extract unique inputs
	if all_inputs is None:
		all_inputs = db['Input'].unique()

	list_inputs_i = [db[db['ID'] == i]['Input'].values for i in db['ID'].unique()]
	list_outputs_i = [db[db['ID'] == i]['Output'].values for i in db['ID'].unique()]

	# Compute all inverse covariance matrices
	inv_0 = kern_0.inverse_covariance_matrix(all_inputs, pen_diag=pen_diag)
	list_inv_i = [kern_i.inverse_covariance_matrix(db[db['ID'] == i]['Input'].values, pen_diag=pen_diag)
	              for i in db['ID'].unique()]

	# Update posterior inverse covariance
	post_inv = inv_0.copy()
	for inputs_i, inv_i in zip(list_inputs_i, list_inv_i):
		# Get indices where individual inputs appear in all_inputs
		indices = lab.array([lab.where(all_inputs == x)[0][0] for x in inputs_i])

		# Create view of the sub-matrix to be updated
		idx_grid = lab.ix_(indices, indices)

		# Update the sub-matrix
		post_inv[idx_grid] += inv_i

	solved = False
	post_cov = None
	while not solved:
		try:
			lower = lab.linalg.cholesky(post_inv + pen_diag)
			post_cov = lab.linalg.inv(lower.T) @ lab.linalg.inv(lower)
			solved = True
		except lab.linalg.LinAlgError:
			logging.WARN(f"Cholesky decomposition failed. Adding more jitter ({pen_diag * 10}) to the diagonal.")
			pen_diag *= 10

	# Update the posterior mean
	weighted_0 = (inv_0 @ m_0).flatten()
	for inputs_i, inv_i, out_i in zip(list_inputs_i, list_inv_i, list_outputs_i):
		weighted_i = inv_i @ out_i

		# Get indices where individual inputs appear in all_inputs
		indices = lab.array([lab.where(all_inputs == x)[0][0] for x in inputs_i])

		# Create view of the sub-matrix to be updated
		idx_grid = lab.ix_(indices)

		# Update the sub-matrix
		weighted_0[idx_grid] += weighted_i

	post_mean = post_cov @ weighted_0

	return post_mean, post_cov


def m_step(db: pd.DataFrame,
		   m_0: lab.array.type,
		   kern_0: Kernel,
		   kern_i: Kernel,
		   post_mean: lab.array.type,
		   post_cov: lab.array.type,
		   common_hp: bool = False,
		   pen_diag: float = config["pen_diag"],
		   all_ids: lab.array.type = None) -> Tuple[lab.array.type, lab.array.type]:
	"""
	M-Step of the EM algorithm

	Maximisation step of the EM algorithm to compute hyperparameters of all the
	kernels involved in Magma.

	:param db: A pandas DataFrame. Columns required: ID, Input, Output.
		Additional columns for covariates can be specified.
	:type db: pd.DataFrame
	:param m_0: A numpy array, corresponding to the prior mean of the mean GP.
	:type m_0: np.ndarray
	:param kern_0: A kernel function, associated with the mean GP.
	:type kern_0: Callable
	:param kern_i: A kernel function, associated with the individual GPs.
	:type kern_i: Callable
	:param post_mean: A numpy array, corresponding to the posterior mean of the mean GP.
	:type post_mean: np.ndarray
	:param post_cov: A numpy array, corresponding to the posterior covariance of the mean GP.
	:type post_cov: np.ndarray
	:param common_hp: A boolean. Whether to use the same hyperparameters for the mean and individual GPs.
		Default is False.
	:type common_hp: bool
	:param pen_diag: A float. A jitter term, added on the diagonal to prevent
		numerical issues when inverting nearly singular matrices. Default is 1e-10.
	:type pen_diag: float
	:param all_ids: A lab array, containing all the distinct Input values
		in the `db` argument.
	:type all_ids: lab.array.type
	:return: A tuple containing the updated hyperparameters of the mean and individual GPs.
	:rtype: Tuple[lab.array.type, lab.array.type]

	:keywords: internal

	:examples: TRUE
	"""
	# Extract unique inputs
	if all_ids is None:
		all_ids = db['Input'].unique()

	# log_likelihood_gp_mod(db, m_0.flatten(), kern_0, post_cov, pen_diag)

	def objective_function(params):
		"""
		Objective function to be minimized by SciPy's optimize.minimize().
		We wrap it here so that scipy can optimise with respect to params even though they are handled as properties of
		the kernel object.
		"""
		from MagmaClustPy.kernels import SquaredExponentialMagmaKernel

		kernel = SquaredExponentialMagmaKernel(*params)
		return -log_likelihood_gp_mod(db, m_0.flatten(), kernel, post_cov, pen_diag)

	# Optimise hyperparameters of the mean process
	result = minimize(
		fun=objective_function,
		x0=lab.array(list(kern_0.params.values())),
		method="L-BFGS-B",
		# TODO: Provide gradient function and see if that accelerates stuff?
		options={'ftol': 1e-13, 'maxiter': 25}
	)

	print(result.x)

	print("hey")
	# TODO: the rest
