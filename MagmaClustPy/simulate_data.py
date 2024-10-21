from typing import Tuple

import numpy as np
import pandas as pd

from MagmaClustPy import lin_alg_backend as lab
from MagmaClustPy.kernels import SquaredExponentialKernel


def simu_db(m: int = 10, n: int = 10, k: int = 1, covariate: bool = False,
            grid: lab.array_type = lab.range(0, 10, 0.05),
            grid_cov: lab.array_type = lab.range(0, 10, 0.5),
            common_input: bool = True, common_hp: bool = True, add_hp: bool = False, add_clust: bool = False,
            int_mu_v: Tuple[int, int] = (4, 5), int_mu_l: Tuple[int, int] = (0, 1), int_i_v: Tuple[int, int] = (1, 2),
            int_i_l: Tuple[int, int] = (0, 1), int_i_sigma: Tuple[float, float] = (0, 0.2),
            lambda_int: Tuple[int, int] = (30, 40), m_int: Tuple[int, int] = (0, 10),
            lengthscale_int: Tuple[int, int] = (30, 40), m0_slope: Tuple[int, int] = (-5, 5),
            m0_intercept: Tuple[int, int] = (-50, 50)) -> lab.array_type:
	"""
	Simulate a dataset tailored for MagmaClustPy.

	Simulate a complete training dataset, which may be representative of various
	applications. Several flexible arguments allow adjustment of the number of
	individuals, observed inputs, and the values of many parameters controlling
	the data generation.

	:param m: The number of individuals per cluster. Default is 10.
	:type m: int, optional
	:param n: The number of observations per individual. Default is 10.
	:type n: int, optional
	:param k: The number of underlying clusters. Default is 1.
	:type k: int, optional
	:param covariate: Indicates whether the dataset should include an additional
		input covariate named 'Covariate'. Default is False.
	:type covariate: bool, optional
	:param grid: A vector of numbers defining a grid of observations
		(i.e., the reference inputs). Default is `np.arange(0, 10, 0.05)`.
	:type grid: numpy.ndarray, optional
	:param grid_cov: A vector of numbers defining a grid of observations
		(i.e., the covariate reference inputs). Default is `np.arange(0, 10, 0.5)`.
	:type grid_cov: numpy.ndarray, optional
	:param common_input: Indicates whether the reference inputs are common to all
		individuals. Default is True.
	:type common_input: bool, optional
	:param common_hp: Indicates whether the hyperparameters are common to all
		individuals. If True and `k > 1`, the hyperparameters remain different
		between the clusters. Default is True.
	:type common_hp: bool, optional
	:param add_hp: Indicates whether the values of hyperparameters should be added
		as columns in the dataset. Default is False.
	:type add_hp: bool, optional
	:param add_clust: Indicates whether the name of the clusters should be added
		as a column in the dataset. Default is False.
	:type add_clust: bool, optional
	:param int_mu_v: An interval of admissible values for the variance hyperparameter
		of the mean process' kernel. Default is (4, 5).
	:type int_mu_v: tuple of int, optional
	:param int_mu_l: An interval of admissible values for the lengthscale hyperparameter
		of the mean process' kernel. Default is (0, 1).
	:type int_mu_l: tuple of int, optional
	:param int_i_v: An interval of admissible values for the variance hyperparameter
		of the individual process' kernel. Default is (1, 2).
	:type int_i_v: tuple of int, optional
	:param int_i_l: An interval of admissible values for the lengthscale hyperparameter
		of the individual process' kernel. Default is (0, 1).
	:type int_i_l: tuple of int, optional
	:param int_i_sigma: An interval of admissible values for the noise hyperparameter.
		Default is (0, 0.2).
	:type int_i_sigma: tuple of float, optional
	:param lambda_int: An interval of admissible values for the lambda parameter of the
		2D exponential. Default is (30, 40).
	:type lambda_int: tuple of int, optional
	:param m_int: An interval of admissible values for the mean of the 2D exponential.
		Default is (0, 10).
	:type m_int: tuple of int, optional
	:param lengthscale_int: An interval of admissible values for the lengthscale parameter
		of the 2D exponential. Default is (30, 40).
	:type lengthscale_int: tuple of int, optional
	:param m0_slope: An interval of admissible values for the slope of `m0`.
		Default is (-5, 5).
	:type m0_slope: tuple of int, optional
	:param m0_intercept: An interval of admissible values for the intercept of `m0`.
		Default is (-50, 50).
	:type m0_intercept: tuple of int, optional

	:returns: A full dataset of simulated training data.
	:rtype: numpy.ndarray

	:examples:

		Generate a dataset with 3 clusters of 4 individuals, observed at 10 inputs::

			>>> data = simu_db(m=4, n=10, k=3)

		Generate a 2-D dataset with an additional input 'Covariate'::

			>>> data = simu_db(covariate=True)

		Generate a dataset where input locations are different among individuals::

			>>> data = simu_db(common_input=False)

		Generate a dataset with an additional column indicating the true clusters::

			>>> data = simu_db(k=3, add_clust=True)
	"""
	if covariate:
		# TODO
		raise NotImplementedError("The 'covariate' argument is not yet implemented.")
	else:
		if common_input:
			t_i = lab.sort(lab.sample(grid, n, replace=False))  # The selected inputs, shared across tasks

		# Dataframe to store the simulated data
		db = pd.DataFrame(columns=['ID', 'Input', 'Output'])

		# Generate clusters
		for cluster in range(1, k + 1):
			# Generate mean process for this cluster
			m_0 = lab.draw(m0_intercept) + lab.draw(m0_slope) * grid  # Mean prior with random slope and intercept
			mu_v = lab.draw(int_mu_v)  # Variance hyperparameter of the mean process' kernel
			mu_l = lab.draw(int_mu_l)  # Lengthscale hyperparameter of the mean process' kernel

			db_0 = simu_indiv_se(_id="0", _input=grid, mean=m_0, v=mu_v, l=mu_l, sigma=0)

			if common_hp:
				i_v = lab.draw(int_i_v)
				i_l = lab.draw(int_i_l)
				i_sigma = lab.draw(int_i_sigma)

			for indiv in range(1, m + 1):
				# Generate individual process
				if not common_input:
					t_i = lab.sort(lab.sample(grid, n, replace=False))

				if not common_hp:
					i_v = lab.draw(int_i_v)
					i_l = lab.draw(int_i_l)
					i_sigma = lab.draw(int_i_sigma)

				# Extract mean for this individual
				mean_i = db_0[db_0['Input'].isin(t_i)]['Output'].values

				_id = f"{indiv}" if k == 1 else f"ID{indiv}-Clust{cluster}"

				# Simulate individual's data
				db_i = simu_indiv_se(_id=_id, _input=t_i, mean=mean_i, v=i_v, l=i_l, sigma=i_sigma)

				if not add_hp:
					db_i = db_i.drop(columns=['se_variance', 'se_lengthscale', 'noise'])

				if add_clust:
					db_i['Cluster'] = cluster

				# Append individual's data to the dataset
				db = pd.concat([db, db_i], ignore_index=True)

	return db


def simu_indiv_se(_id: str, _input: lab.array_type, mean: lab.array_type, v: float, l: float, sigma: float) \
		-> pd.DataFrame:
	"""
	Simulate a batch of data for one individual using a GP with the Squared Exponential kernel.

	Simulate a batch of output data, corresponding to one individual, coming from
	a GP with the Squared Exponential kernel as covariance structure, and
	specified hyper-parameters and input.

	:param _id: An identification code, whether numeric or character.
	:type _id: str
	:param _input: A vector of numbers. The input variable that is used as
		'reference' for input and outputs.
	:type _input: numpy.ndarray
	:param mean: A vector of numbers. Prior mean values of the GP.
	:type mean: numpy.ndarray
	:param v: The variance hyper-parameter of the SE kernel.
	:type v: float
	:param l: The lengthscale hyper-parameter of the SE kernel.
	:type l: float
	:param sigma: The noise hyper-parameter.
	:type sigma: float

	:return: A DataFrame containing a batch of output data along with input and
		additional information for a simulated individual.
	:rtype: pandas.DataFrame

	:examples:

		TRUE
	"""
	cov = SquaredExponentialKernel(length_scale=l, variance=v).compute_matrix(_input)

	# Add noise to the covariance matrix
	cov += np.diag([sigma] * len(mean))

	# Generate the multivariate normal random vector
	output = np.random.multivariate_normal(mean, cov)

	# Create a DataFrame with the simulated data
	db = pd.DataFrame(
		{'ID': _id, 'Input': _input, 'Output': output, 'se_variance': v, 'se_lengthscale': l, 'noise': sigma})
	# TODO (enhancement): naming convention for columns, why are some uppercase and some lowercase?

	return db
