from typing import Any, Dict, Callable
import logging
import time

import pandas as pd

from MagmaClustPy import lin_alg_backend as lab
from MagmaClustPy import config
from MagmaClustPy.kernels import Kernel, SquaredExponentialKernel
from MagmaClustPy.em_magma import e_step, m_step


def train_magma(
		data: pd.DataFrame,
		prior_mean: float | lab.array.type | Callable | pd.DataFrame = None,
		kern_0: Kernel = None,
		kern_i: Kernel = None,
		common_hp: bool = True,
		grid_inputs: lab.array.type = None,
		pen_diag: float = config["pen_diag"],
		n_iter_max: int = 25,
		cv_threshold: float = 1e-3,
		fast_approx: bool = False
) -> (pd.DataFrame, pd.DataFrame, Dict[str, Any], lab.array.type, lab.array.type, bool, float):
	"""
	Train Magma with an EM algorithm.

	The hyperparameters and the hyper-posterior distribution involved in Magma
	can be learned thanks to an EM algorithm implemented in `train_magma`.
	By providing a dataset, the model hypotheses (hyper-prior mean parameter and
	covariance kernels) and initialisation values for the hyperparameters, the
	function computes maximum likelihood estimates of the HPs as well as the
	mean and covariance parameters of the Gaussian hyper-posterior distribution
	of the mean process.

	:param data: A pandas DataFrame. Required columns: `ID`, `Input`, `Output`.
		Additional columns for covariates can be specified. The `ID` column contains
		the unique names/codes used to identify each individual/task (or batch of data).
		The `Input` column should define the variable that is used as reference for the
		observations (e.g. time for longitudinal data). The `Output` column specifies
		the observed values (the response variable). The data frame can also provide
		as many covariates as desired, with no constraints on the column names. These
		covariates are additional inputs (explanatory variables) of the models that are
		also observed at each reference `Input`.
	:type data: pd.DataFrame
	:param prior_mean: Hyper-prior mean parameter (m_0) of the mean GP. This argument
		can be specified under various formats, such as:
		- None (default). The hyper-prior mean would be set to 0 everywhere.
		- A number. The hyper-prior mean would be a constant function.
		- A vector of the same length as all the distinct Input values in the `data` argument.
		  This vector would be considered as the evaluation of the hyper-prior mean function
		  at the training Inputs.
		- A function. This function is defined as the hyper_prior mean.
		- A pandas DataFrame. Required columns: Input, Output. The Input values should include
		  at least the same values as in the `data` argument.
	:type prior_mean: float | lab.array.type | Callable | pd.DataFrame
	:param kern_0: A Kernel instance, associated with the mean GP. Several popular kernels
		(see `The Kernel Cookbook <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_) are already implemented
		and can be selected within the following list:
		- SquaredExponentialKernel (also called Radial Basis Function or Gaussian kernel) (default)
		- TODO LinearKernel
		- TODO PeriodicKernel
		- TODO RationalQuadraticKernel
		Compound kernels can be created as sums or products of the above kernels.
		N.b: the kernel hyperparameters will be modified during training.
	:type kern_0: Kernel
	:param kern_i: if `common_hp == True`, a common Kernel instance, associated with the individual GPs. If
		`common_hp == False`, this argument can either be a Kernel instance or a dictionary of Kernel instances.
		In the first case, the kernel will be cloned for each individual. In the second case, the dictionary
		must contain a key for each individual ID, and the corresponding value must be a Kernel instance.
		N.b: the kernel(s) hyperparameters will be modified during training.
	:type kern_i: Kernel | Dict[str, Kernel]
	:param common_hp: A logical value, indicating whether the set of hyperparameters is assumed to be common to all
		individuals.
	:type common_hp: bool
	:param grid_inputs: A vector, indicating the grid of additional reference inputs on which the mean process'
		hyper-posterior should be evaluated.
	:type grid_inputs: lab.array.type
	:param pen_diag: A number. A jitter term, added on the diagonal to prevent numerical issues when inverting nearly
		singular matrices.
	:type pen_diag: float
	:param n_iter_max: A number, indicating the maximum number of iterations of the EM algorithm to proceed while not
		reaching convergence.
	:type n_iter_max: int
	:param cv_threshold: A number, indicating the threshold of the likelihood gain under which the EM algorithm will
		stop. The convergence condition is defined as the difference of likelihoods between two consecutive steps,
		divided by the absolute value of the last one ( (LL_n - LL_n-1) / |LL_n| ).
	:type cv_threshold: float
	:param fast_approx: A boolean, indicating whether the EM algorithm should stop after only one iteration of the
		E-step. This advanced feature is mainly used to provide a faster approximation of the model selection procedure,
		by preventing any optimisation over the hyperparameters.
	:type fast_approx: bool

	:return: A tuple, gathering the results of the EM algorithm used for training in Magma. The elements of the tuple
		are:
		- hp_0: A pandas DataFrame of the trained hyperparameters for the mean process' kernel.
		- hp_i: A pandas DataFrame of all the trained hyperparameters for the individual processes' kernels.
		- hyperpost: A sub-dictionary gathering the parameters of the mean processes' hyper-posterior distributions,
			namely:
			- mean: A pandas DataFrame, the hyper-posterior mean parameter (`Output`) evaluated at each training
				reference `Input`.
			- cov: An array, the covariance parameter for the hyper-posterior distribution of the mean process.
			- pred: A pandas DataFrame, the predicted mean and variance at `Input` for the mean process' hyper-posterior
				distribution under a format that allows the direct visualisation as a GP prediction.
		- ini_args: A dictionary containing the initial function arguments and values for the hyper-prior mean, the
			hyperparameters. In particular, if those arguments were set to None, `ini_args` allows us to retrieve the
			(randomly chosen) initialisations used  during training.
		- seq_loglikelihood: An array, containing the sequence of log-likelihood values associated with each iteration.
		- converged: A boolean value indicated whether the EM algorithm converged or not.
		- training_time: Total running time of the complete training.
	:rtype: (pd.DataFrame, pd.DataFrame, Dict[str, Any], lab.array.type, lab.array.type, bool, float)
	"""
	# Asser the correct type and format of the data
	assert isinstance(data, pd.DataFrame), "The data argument must be a pandas DataFrame."
	assert "ID" in data.columns, "The data argument must contain a column named 'ID'."
	assert "Input" in data.columns, "The data argument must contain a column named 'Input'."
	assert "Output" in data.columns, "The data argument must contain a column named 'Output'."

	# Track training time
	start_time = time.time()

	# If additional covariates are provided, warn that this is not yet supported
	if len(data.columns) > 3:
		print("Warning: Additional covariates are not yet supported and will be ignored.")
		data = data[["ID", "Input", "Output"]]

	# Drop NaN values
	data = data.dropna()

	# Extract unique IDs
	all_ids = data["ID"].unique()  # FIXME: this returns a numpy array whatever the lab backend is

	# Extract all the distinct Input values
	all_inputs = data["Input"].unique()  # FIXME: this returns a numpy array whatever the lab backend is
	all_inputs.sort()  # TODO: remove sort()

	# Initialise prior mean
	if prior_mean is None:
		prior_mean = lab.zeros((len(all_inputs), 1))
		logging.info("The 'prior_mean' argument has not been specified. The hyper_prior mean function is thus set to "
		             "be 0 everywhere.\n\n")
	elif isinstance(prior_mean, (float, int)):
		prior_mean = lab.zeros((len(all_inputs), 1)) + prior_mean
		logging.info("The provided 'prior_mean' argument is of length 1. Thus, the hyper_prior mean function has set to"
		             " be constant everywhere.\n\n")
	else:
		# TODO: Actually implement.
		raise NotImplementedError("The 'prior_mean' argument as an array, DataFrame or function is not yet supported.")

	if kern_0 is None:
		kern_0 = SquaredExponentialKernel()
		logging.info("The 'kern_0' argument has not been specified. The mean GP kernel is thus set to be the "
		             "Squared Exponential Kernel with random initialisation.\n\n")

	if kern_i is None:
		kern_i = SquaredExponentialKernel()
		logging.info("The 'kern_i' argument has not been specified. The individual GP kernel is thus set to be the "
		             "Squared Exponential Kernel with random initialisation.\n\n")

	# TODO: add history of hyperparameters
	converged = False
	current_log_likelihood = -float("inf")
	log_likelihoods_history = []

	# Main training loop
	for i in range(1, n_iter_max+1):
		# Track time for each iteration
		i_start_time = time.time()

		# E-step
		post_mean, post_cov = e_step(db=data, m_0=prior_mean, kern_0=kern_0, kern_i=kern_i, pen_diag=pen_diag, all_inputs=all_inputs)

		# Break after E-step if we can compute the fast approximation
		if fast_approx:
			# TODO: implement fast approximation
			raise NotImplementedError("The 'fast_approx' argument is not yet supported.")

		hp_0, hp_i = m_step(db=data, m_0=prior_mean, kern_0=kern_0, kern_i=kern_i, post_mean=post_mean, post_cov=post_cov, common_hp=common_hp, pen_diag=pen_diag, all_ids=all_ids)
		print(post_mean, post_cov)
		# TODO: next steps of the algorithm







