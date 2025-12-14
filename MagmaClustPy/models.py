# Standard library
from abc import ABC, abstractmethod
import logging

# Third party
from jax import vmap
from jax import numpy as jnp
import numpy as np
import pandas as pd
from kernax import AbstractKernel, BatchKernel

# Local
from MagmaClustPy.utils import preprocess_db, check_db
from MagmaClustPy.linalg import compute_mapping, lexicographic_sort
from MagmaClustPy.hyperpost import hyperpost
from MagmaClustPy.hp_optimisation import optimise_mean_kernel, optimise_task_kernel
from MagmaClustPy.prediction import predict
from MagmaClustPy.mixture import update_mixture
from MagmaClustPy.initialisation import init_mixture
from MagmaClustPy.means import BasePriorMean


class BaseLikelihood:
	# TODO: move into own script
	pass


class BaseModel(ABC):
	@abstractmethod
	def load_train_data(self, db: pd.DataFrame):
		"""
		Loads training data from a database, populating train attributes.
		The training data is used to fit the model, aka learn the hyperparameters, mean process, etc.


		:param db: pandas DataFrame with columns "Task_ID", "Input", "Input_ID", "Output", "Output_ID"
		:return:
		"""
		raise NotImplementedError

	@abstractmethod
	def load_pred_data(self, db):
		"""
		Loads pred data from a database, populating pred attributes.
		The pred data is used to make predictions after fitting the model. The model is conditioned on the training data *and* the pred data.

		:param db:
		:return:
		"""
		raise NotImplementedError

	@abstractmethod
	def load_test_data(self, db: pd.DataFrame):
		"""
		Loads test data from a database, populating test attributes.
		The test data is used to compare predictions from the model (after fitting and conditioning on pred data) against ground truth values.
		Each ID in test data must coincide with an ID in pred data. Test data contains the points where outputs are known but we want to hide them from the model during fitting and prediction.

		:param db: pandas DataFrame with columns "Task_ID", "Input", "Input_ID", "Output", "Output_ID"
		:return:
		"""
		raise NotImplementedError

	@abstractmethod
	def fit(self):
		"""
		Fits the model to the training data.

		:return:
		"""
		raise NotImplementedError

	@abstractmethod
	def predict(self, X_test):
		"""
		Makes predictions on test data.

		:param X_test:
		:return:
		"""
		raise NotImplementedError

	@abstractmethod
	def plot_mean_process(self):
		"""
		Plots mean process.

		:return:
		"""
		raise NotImplementedError

	@abstractmethod
	def plot_predictions(self):
		"""
		Plots predictions.

		:return:
		"""
		raise NotImplementedError


class Magma(BaseModel):
	# TODO: make Magma a special case of MagmaClust where k=1
	def __init__(self,
	             likelihood: BaseLikelihood,
	             prior_mean: BasePriorMean,
	             mean_kernel: AbstractKernel,
	             task_kernel_train: AbstractKernel,
	             task_kernel_pred: AbstractKernel,
	             shared_hp: bool):
		self.likelihood = likelihood
		self.prior_mean = prior_mean
		self.mean_kernel = mean_kernel
		self.task_kernel_train = task_kernel_train
		self.task_kernel_pred = task_kernel_pred
		self.shared_hp = shared_hp

		# Attributes that will be instantiated later
		self.padded_inputs_train = None
		self.padded_outputs_train = None
		self.mappings_train = None
		self.all_inputs_train = None
		self.shared_inputs_train = None

		self.padded_inputs_pred = None
		self.padded_outputs_pred = None
		self.mappings_pred = None
		self.all_inputs_pred = None

		self.padded_inputs_test = None
		self.padded_outputs_test = None
		self.mappings_test = None
		self.all_inputs_test = None

		self.post_mean = None
		self.post_cov = None

	def load_train_data(self, db: pd.DataFrame, skip_check=False):
		if not skip_check:
			check_db(db)
		self.padded_inputs_train, self.padded_outputs_train, self.mappings_train, self.all_inputs_train = preprocess_db(
			db)
		self.shared_inputs_train = self.padded_inputs_train[0].shape == self.all_inputs_train.shape and jnp.all(self.padded_inputs_train[0] == self.all_inputs_train).item()

		# Batch kernels, if they are not already batched
		if not isinstance(self.task_kernel_train, BatchKernel):
			if self.shared_hp:
				self.task_kernel_train = BatchKernel(self.task_kernel_train,
				                          batch_size=self.padded_inputs_train.shape[0], batch_in_axes=None, batch_over_inputs=True)
			else:
				self.task_kernel_train = BatchKernel(self.task_kernel_train,
				                          batch_size=self.padded_inputs_train.shape[0], batch_in_axes=0, batch_over_inputs=True)

	def load_pred_data(self, db: pd.DataFrame, skip_check=True):
		if not skip_check:
			check_db(db)
		self.padded_inputs_pred, self.padded_outputs_pred, self.mappings_pred, self.all_inputs_pred = preprocess_db(db)

		if not isinstance(self.task_kernel_pred, BatchKernel):
			if self.shared_hp:
				self.task_kernel_pred = BatchKernel(self.task_kernel_pred,
				                          batch_size=self.padded_inputs_pred.shape[0], batch_in_axes=None, batch_over_inputs=True)
			else:
				self.task_kernel_pred = BatchKernel(self.task_kernel_pred,
				                          batch_size=self.padded_inputs_pred.shape[0], batch_in_axes=0, batch_over_inputs=True)

	def load_test_data(self, db: pd.DataFrame, skip_check=True):
		if not skip_check:
			check_db(db)
		self.padded_inputs_test, self.padded_outputs_test, self.mappings_test, all_inputs_test = preprocess_db(db)

	def fit(self, max_iter: int = 25, converg_threshold: float = 1e-3, jitter: jnp.ndarray = jnp.array(1e-4)):
		# Monitoring variables
		prev_mean_llh = jnp.inf
		prev_task_llh = jnp.inf
		conv_ratio = jnp.inf

		for i in range(max_iter):
			logging.info(
				f"Iteration {i:4}\tLlhs: {prev_mean_llh:12.4f}, {prev_task_llh:12.4f}\tConv. Ratio: {conv_ratio:.5f}\t\n\tMean kernel: {self.mean_kernel}\n\tTask kernel: {self.task_kernel_train}")
			# e-step: compute hyper-posterior
			prior_mean_on_grid = self.prior_mean(self.all_inputs_train)
			self.post_mean, self.post_cov = hyperpost(self.padded_inputs_train, self.padded_outputs_train, self.mappings_train,
			                                self.all_inputs_train,
			                                prior_mean_on_grid, self.mean_kernel, self.task_kernel_train,
			                                shared_input=self.shared_inputs_train, shared_hp=self.shared_hp)

			# m-step: update hyperparameters
			self.mean_kernel, mean_llh = optimise_mean_kernel(self.mean_kernel, self.all_inputs_train, prior_mean_on_grid,
			                                             self.post_mean, self.post_cov, jitter=jitter)
			self.task_kernel_train, task_llh = optimise_task_kernel(self.task_kernel_train, self.padded_inputs_train, self.padded_outputs_train,
			                                                        self.mappings_train, self.post_mean[None, :], self.post_cov[None, :, :],
			                                                        shared_hp=self.shared_hp, cluster_hp=False,jitter=jitter)

			# Check for NaN values and stop early
			if jnp.isnan(mean_llh) or jnp.isnan(task_llh):
				logging.error(f"NaN detected at iteration {i}. Stopping training.")
				break

			# Check convergence
			if i > 0:
				conv_ratio = jnp.abs((prev_mean_llh + prev_task_llh) - (mean_llh + task_llh)) / jnp.abs(
					prev_mean_llh + prev_task_llh)
			if conv_ratio < converg_threshold:
				logging.info(
					f"Convergence reached after {i + 1} iterations.\tNLLs: {mean_llh:12.4f}, {task_llh:12.4f}\n\tMean kernel: {self.mean_kernel}\n\tTask kernel: {self.task_kernel_train}")
				break

			if i == max_iter - 1:
				logging.warning(
					f"Maximum number of iterations reached.\nLast modif: {jnp.abs(prev_mean_llh - mean_llh).item()} & {jnp.abs(prev_task_llh - task_llh).item()}")

			prev_mean_llh = mean_llh
			prev_task_llh = task_llh

	def optimise_pred_kernels(self, jitter: jnp.ndarray = jnp.array(1e-4)):
		# Optimise the task kernel for prediction
		self.task_kernel_pred, _ = optimise_task_kernel(self.task_kernel_pred, self.padded_inputs_pred, self.padded_outputs_pred,
		                                                self.mappings_pred, self.post_mean[None, :], self.post_cov[None, :, :],
		                                                shared_hp=self.shared_hp, cluster_hp=False, jitter=jitter)

	def predict(self, grid: np.ndarray, skip_retrain: bool=False) -> np.ndarray:
		if not self.shared_hp and not skip_retrain:
			self.optimise_pred_kernels()

		# Merge grid and all_inputs and compute new mappings
		full_grid = lexicographic_sort(jnp.unique(jnp.concatenate([self.all_inputs_train, self.all_inputs_pred, grid]), axis=0))
		# Compute new mappings
		mappings_train_on_grid = vmap(compute_mapping, in_axes=(None, 0))(full_grid, self.padded_inputs_train)
		mappings_pred_on_grid = vmap(compute_mapping, in_axes=(None, 0))(full_grid, self.padded_inputs_pred)

		# Compute the hyper-posterior on the grid
		post_mean_grid, post_cov_grid = hyperpost(inputs=self.padded_inputs_train,
		                                          outputs=self.padded_outputs_train,
		                                          mappings=mappings_train_on_grid,
		                                          all_inputs=full_grid,
		                                          prior_mean=jnp.array(0.),
		                                          mean_kernel=self.mean_kernel,
		                                          task_kernel=self.task_kernel_train,
		                                          shared_input=False,  # As we use a grid
		                                          shared_hp=self.shared_hp)

		# Compute predictions
		return predict(post_mean_grid, post_cov_grid, self.padded_outputs_pred, mappings_pred_on_grid, full_grid, self.task_kernel_pred)

	def plot_predictions(self):
		pass

	def plot_mean_process(self):
		pass

	def generate_grid(self, grid_size, margin=5):
		return jnp.linspace(jnp.min(self.all_inputs_train - margin, axis=0), jnp.max(self.all_inputs_train + margin, axis=0), grid_size)


class MagmaClust(BaseModel):
	def __init__(self,
	             k: int,
	             likelihood: BaseLikelihood,
	             prior_mean: BasePriorMean,
	             mean_kernel: AbstractKernel,
	             task_kernel_train: AbstractKernel,
	             task_kernel_pred: AbstractKernel,
	             shared_hp: bool,
	             cluster_hp: bool):
		self.k = k
		self.likelihood = likelihood
		self.prior_mean = prior_mean
		self.mean_kernel = mean_kernel
		self.task_kernel_train = task_kernel_train
		self.task_kernel_pred = task_kernel_pred
		self.shared_hp = shared_hp
		self.cluster_hp = cluster_hp

		# Attributes that will be instantiated later
		self.padded_inputs_train = None
		self.padded_outputs_train = None
		self.mappings_train = None
		self.all_inputs_train = None
		self.shared_inputs_train = None

		self.padded_inputs_pred = None
		self.padded_outputs_pred = None
		self.mappings_pred = None
		self.all_inputs_pred = None

		self.padded_inputs_test = None
		self.padded_outputs_test = None
		self.mappings_test = None
		self.all_inputs_test = None

		self.post_means = None
		self.post_covs = None

		self.mixture_train = None
		self.mixture_pred = None

	def batch_kernel(self, kernel, nb_tasks, nb_clusters):
		if self.shared_hp and not self.cluster_hp:
			# Batch along tasks
			kernel = BatchKernel(kernel, batch_size=nb_tasks, batch_in_axes=None, batch_over_inputs=True)
		elif self.shared_hp and self.cluster_hp:
			# Batch along tasks
			kernel = BatchKernel(kernel, batch_size=nb_tasks, batch_in_axes=None, batch_over_inputs=True)

			# Batch along clusters
			kernel = BatchKernel(kernel, batch_size=nb_clusters, batch_in_axes=0, batch_over_inputs=False)
		elif not self.shared_hp and not self.cluster_hp:
			# Batch along tasks
			kernel = BatchKernel(kernel, batch_size=nb_tasks, batch_in_axes=0, batch_over_inputs=True)
		else:  # not shared_hp and cluster_hp
			# Batch along tasks
			kernel = BatchKernel(kernel, batch_size=nb_tasks, batch_in_axes=0, batch_over_inputs=True)

			# Batch along clusters
			kernel = BatchKernel(kernel, batch_size=nb_clusters, batch_in_axes=0, batch_over_inputs=False)

		return kernel

	def load_train_data(self, db: pd.DataFrame, skip_check=False):
		if not skip_check:
			check_db(db)
		self.padded_inputs_train, self.padded_outputs_train, self.mappings_train, self.all_inputs_train = preprocess_db(
			db)
		self.shared_inputs_train = self.padded_inputs_train[0].shape == self.all_inputs_train.shape and jnp.all(self.padded_inputs_train[0] == self.all_inputs_train).item()

		# Batch kernels, if they are not already batched
		if not isinstance(self.task_kernel_train, BatchKernel):
			self.task_kernel_train = self.batch_kernel(self.task_kernel_train, self.padded_inputs_train.shape[0], self.k)

		if self.k == 1:
			# No clustering, so no need for cluster_hp and mixture
			self.cluster_hp = False
			self.mixture_train = jnp.ones((1, self.padded_inputs_train.shape[0]))


	def load_pred_data(self, db: pd.DataFrame, skip_check=True):
		if not skip_check:
			check_db(db)
		self.padded_inputs_pred, self.padded_outputs_pred, self.mappings_pred, self.all_inputs_pred = preprocess_db(db)

		if not isinstance(self.task_kernel_pred, BatchKernel):
			self.task_kernel_pred = self.batch_kernel(self.task_kernel_pred, self.padded_inputs_pred.shape[0], self.k)

		if self.k == 1:
			# No clustering, so no need for mixture
			self.mixture_pred = jnp.ones((1, self.padded_inputs_pred.shape[0]))

	def load_test_data(self, db: pd.DataFrame, skip_check=True):
		if not skip_check:
			check_db(db)
		self.padded_inputs_test, self.padded_outputs_test, self.mappings_test, all_inputs_test = preprocess_db(db)

	def fit(self, max_iter: int = 25, converg_threshold: float = 1e-3, jitter: jnp.ndarray = jnp.array(1e-4)):
		# Monitoring variables
		prev_mean_llh = jnp.inf
		prev_task_llh = jnp.inf
		conv_ratio = jnp.inf

		if self.mixture_train is None:
			# Initialise mixture with k-means
			self.mixture_train = init_mixture(self.padded_outputs_train, self.k, self.shared_hp)

		for i in range(max_iter):
			logging.info(
				f"Iteration {i:4}\tLlhs: {prev_mean_llh:12.4f}, {prev_task_llh:12.4f}\tConv. Ratio: {conv_ratio:.5f}\t\n\tMean kernel: {self.mean_kernel}\n\tTask kernel: {self.task_kernel_train}")

			# e-step: compute hyper-posterior
			prior_mean_on_grid = self.prior_mean(self.all_inputs_train)
			if self.cluster_hp:
				batched_hyperpost = vmap(hyperpost, in_axes=(None, None, None, None, None, None, self.task_kernel_train.batch_in_axes, None, None, 0))
				self.post_means, self.post_covs = batched_hyperpost(self.padded_inputs_train,
				                                                    self.padded_outputs_train,
				                                                    self.mappings_train,
				                                                    self.all_inputs_train, prior_mean_on_grid,
				                                                    self.mean_kernel,
				                                                    self.task_kernel_train.inner_kernel,
				                                                    self.shared_inputs_train,
				                                                    self.shared_hp,
				                                                    self.mixture_train)
			else: # not cluster_hp
				batched_hyperpost = vmap(hyperpost, in_axes=(None, None, None, None, None, None, None, None, None, 0))
				self.post_means, self.post_covs = batched_hyperpost(self.padded_inputs_train,
				                                                    self.padded_outputs_train,
				                                                    self.mappings_train,
				                                                    self.all_inputs_train, prior_mean_on_grid,
				                                                    self.mean_kernel,
				                                                    self.task_kernel_train,
				                                                    self.shared_inputs_train,
				                                                    self.shared_hp,
				                                                    self.mixture_train)

			if self.k > 1:
				# mixture-step: update the mixture using likelihood of each task for each mean process
				self.mixture_train = update_mixture(self.task_kernel_train, self.padded_inputs_train, self.padded_outputs_train, self.mappings_train, self.post_means, self.post_covs, self.shared_hp, self.cluster_hp, jitter=jitter)

			# m-step: update hyperparameters
			self.mean_kernel, mean_llh = optimise_mean_kernel(self.mean_kernel, self.all_inputs_train, prior_mean_on_grid,
			                                                  self.post_means, self.post_covs, jitter=jitter)
			self.task_kernel_train, task_llh = optimise_task_kernel(self.task_kernel_train, self.padded_inputs_train, self.padded_outputs_train,
			                                                self.mappings_train, self.post_means, self.post_covs,
			                                                mixture_coeffs=self.mixture_train, shared_hp=self.shared_hp, cluster_hp=self.cluster_hp, jitter=jitter)

			# Check for NaN values and stop early
			if jnp.isnan(mean_llh) or jnp.isnan(task_llh):
				logging.error(f"NaN detected at iteration {i}. Stopping training.")
				break

			# Check convergence
			if i > 0:
				conv_ratio = jnp.abs((prev_mean_llh + prev_task_llh) - (mean_llh + task_llh)) / jnp.abs(
					prev_mean_llh + prev_task_llh)
			if conv_ratio < converg_threshold:
				logging.info(
					f"Convergence reached after {i + 1} iterations.\tNLLs: {mean_llh:12.4f}, {task_llh:12.4f}\n\tMean kernel: {self.mean_kernel}\n\tTask kernel: {self.task_kernel_train}")
				break

			if i == max_iter - 1:
				logging.warning(
					f"Maximum number of iterations reached.\nLast modif: {jnp.abs(prev_mean_llh - mean_llh).item()} & {jnp.abs(prev_task_llh - task_llh).item()}")

			prev_mean_llh = mean_llh
			prev_task_llh = task_llh

	def optimise_pred_kernels(self, jitter: jnp.ndarray = jnp.array(1e-4)):
		# Optimise the task kernel for prediction
		self.task_kernel_pred, _ = optimise_task_kernel(self.task_kernel_pred, self.padded_inputs_pred, self.padded_outputs_pred,
		                                                self.mappings_pred, self.post_means, self.post_covs,
		                                                mixture_coeffs=self.mixture_train, shared_hp=self.shared_hp,
		                                                cluster_hp=self.cluster_hp, jitter=jitter)

	def predict(self, grid: np.ndarray, skip_retrain: bool=False, jitter: jnp.ndarray = jnp.array(1e-4)) -> np.ndarray:
		if not self.shared_hp and not skip_retrain:
			self.optimise_pred_kernels()

		if self.mixture_pred is None:
			# Set mixture
			self.mixture_pred = update_mixture(
				self.task_kernel_pred, self.padded_inputs_pred, self.padded_outputs_pred, self.mappings_pred,
				self.post_means, self.post_covs,
				self.shared_hp, self.cluster_hp, jitter=jitter)

		# Merge grid and all_inputs and compute new mappings
		full_grid = lexicographic_sort(jnp.unique(jnp.concatenate([self.all_inputs_train, self.all_inputs_pred, grid]), axis=0))
		# Compute new mappings
		mappings_train_on_grid = vmap(compute_mapping, in_axes=(None, 0))(full_grid, self.padded_inputs_train)
		mappings_pred_on_grid = vmap(compute_mapping, in_axes=(None, 0))(full_grid, self.padded_inputs_pred)

		# Compute the hyper-posterior on the grid
		post_mean_grid, post_cov_grid = hyperpost(inputs=self.padded_inputs_train,
		                                          outputs=self.padded_outputs_train,
		                                          mappings=mappings_train_on_grid,
		                                          all_inputs=full_grid,
		                                          prior_mean=jnp.array(0.),
		                                          mean_kernel=self.mean_kernel,
		                                          task_kernel=self.task_kernel_pred,
		                                          shared_input=False,  # As we use a grid
		                                          shared_hp=self.shared_hp)

		# Compute predictions
		return predict(post_mean_grid, post_cov_grid, self.padded_outputs_pred, mappings_pred_on_grid, full_grid, self.task_kernel_pred)

	def plot_predictions(self):
		pass

	def plot_mean_process(self):
		pass

	def generate_grid(self, grid_size, margin=5):
		return jnp.linspace(jnp.min(self.all_inputs_train - margin, axis=0), jnp.max(self.all_inputs_train + margin, axis=0), grid_size)
