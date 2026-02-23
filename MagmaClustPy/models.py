# Standard library
from abc import ABC, abstractmethod
import logging
from typing import Tuple, Literal, Optional

# Third party
from jax import vmap, Array
from jax import numpy as jnp
import jax.random as jr
import pandas as pd
from kernax import AbstractKernel, BatchKernel
import matplotlib.pyplot as plt

# Local
from MagmaClustPy.utils import preprocess_db, check_db
from MagmaClustPy.linalg import compute_mapping, lexicographic_sort
from MagmaClustPy.hyperpost import hyperpost
from MagmaClustPy.hp_optimisation import optimise_mean_kernel, optimise_task_kernel
from MagmaClustPy.prediction import predict_single_cluster
from MagmaClustPy.mixture import update_mixture
from MagmaClustPy.initialisation import init_mixture
from MagmaClustPy.means import BasePriorMean
from MagmaClustPy.plot import plot_tasks, plot_points, plot_process, plot_samples


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

		self.grid_pred = None
		self.post_mean_pred = None
		self.post_cov_pred = None
		self.mean_pred = None
		self.cov_pred = None

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

	def fit(self, max_iter: int = 25, converg_threshold: float = 1e-3, jitter: Array = jnp.array(1e-4), freeze_mean_hps: bool = False, freeze_task_hps: bool = False):
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
			mean_llh = 0
			if not freeze_mean_hps:
				self.mean_kernel, mean_llh = optimise_mean_kernel(self.mean_kernel, self.all_inputs_train, prior_mean_on_grid,
				                                             self.post_mean, self.post_cov, jitter=jitter)
			task_llh = 0
			if not freeze_task_hps:
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
					f"Convergence reached after {i + 1} iterations.\tNLLs: {mean_llh:12.4f}, {task_llh:12.4f}\n\tMean kernel: {str(self.mean_kernel)}\n\tTask kernel: {str(self.task_kernel_train)}")
				break

			if i == max_iter - 1:
				logging.warning(
					f"Maximum number of iterations reached.\nLast modif: {jnp.abs(prev_mean_llh - mean_llh).item()} & {jnp.abs(prev_task_llh - task_llh).item()}")

			prev_mean_llh = mean_llh
			prev_task_llh = task_llh

	def optimise_pred_kernels(self, jitter: Array = jnp.array(1e-4)):
		# Optimise the task kernel for prediction
		self.task_kernel_pred, _ = optimise_task_kernel(self.task_kernel_pred, self.padded_inputs_pred, self.padded_outputs_pred,
		                                                self.mappings_pred, self.post_mean[None, :], self.post_cov[None, :, :],
		                                                shared_hp=self.shared_hp, cluster_hp=False, jitter=jitter)

	def predict(self, skip_retrain: bool = False) -> Tuple[Array]:
		if not self.shared_hp and not skip_retrain:
			self.optimise_pred_kernels()

		# Merge grid and all_inputs
		full_grid = lexicographic_sort(
			jnp.unique(jnp.concatenate([self.all_inputs_train, self.all_inputs_pred, self.grid_pred]), axis=0))
		# Compute new mappings
		mappings_train_on_grid = vmap(compute_mapping, in_axes=(None, 0))(full_grid, self.padded_inputs_train)
		mappings_pred_on_grid = vmap(compute_mapping, in_axes=(None, 0))(full_grid, self.padded_inputs_pred)
		mappings_grid_on_full_grid = vmap(compute_mapping, in_axes=(None, 0))(full_grid, self.grid_pred)

		# Compute the hyper-posterior on the grid
		post_mean_full_grid, post_cov_full_grid = hyperpost(inputs=self.padded_inputs_train,
		                                                    outputs=self.padded_outputs_train,
		                                                    mappings=mappings_train_on_grid,
		                                                    all_inputs=full_grid,
		                                                    prior_mean=jnp.array(0.),
		                                                    mean_kernel=self.mean_kernel,
		                                                    task_kernel=self.task_kernel_train,
		                                                    shared_input=False,  # As we use a grid
		                                                    shared_hp=self.shared_hp)

		self.post_mean_grid = post_mean_full_grid[mappings_grid_on_full_grid]
		self.post_cov_grid = post_cov_full_grid[jnp.ix_(mappings_grid_on_full_grid, mappings_grid_on_full_grid)]

		post_mean_pred = vmap(lambda m: post_mean_full_grid[m])(mappings_pred_on_grid)
		post_cov_pred = vmap(lambda m: post_cov_full_grid[jnp.ix_(m, m)])(mappings_pred_on_grid)
		post_cov_crossed = vmap(lambda m: post_cov_full_grid[jnp.ix_(m, mappings_grid_on_full_grid)])(mappings_pred_on_grid)

		# Compute predictions
		self.mean_pred, self.cov_pred = predict_single_cluster(
			self.padded_inputs_pred,
			self.padded_outputs_pred,
			self.grid_pred,
			post_mean_pred, post_cov_pred, post_cov_crossed,
			self.post_mean_grid, self.post_cov_grid,
			self.task_kernel_pred)

		return self.mean_pred, self.cov_pred

	def sample(self, nb_samples: int, jitter: Array = jnp.array(1e-4), key: Array = jr.PRNGKey(42)) -> Array:
		"""
		Return nb_samples random samples for each task in padded_inputs_pred and padded_outputs_pred.

		:param nb_samples: number of samples to return
		:param jitter: term for numerical stability
		:param key: PRNG key
		:return: array of shape (N, nb_samples, G), with:
			* N: the number of prediction tasks (first dimension of self.padded_inputs_pred)
			* G: the number of points in the prediction grid (last dimension of self.grid_pred)
		"""

		eye = jitter * jnp.eye(len(self.cov_pred[0]))
		sample_task = lambda mean, cov: jr.multivariate_normal(key, mean, cov+eye,
		                              shape=(nb_samples,))
		return vmap(sample_task)(self.mean_pred, self.cov_pred)

	def plot_predictions(self, plot_as: Literal["samples", "process"] = "samples", task_id=0,
	                           nb_samples: int = 100, include_train_points=True, key: Array = jr.PRNGKey(42), fig=None,
	                           ax=None):
		if fig is None and ax is None:
			fig, ax = plt.subplots(figsize=(12, 8))

		if include_train_points:
			plot_tasks(self.padded_inputs_train, self.padded_outputs_train, point_alpha=0.1, fig=fig, ax=ax)

		# Plot prediction
		if plot_as == "samples":
			key, subkey = jr.split(key)
			samples = self.sample(nb_samples, key=subkey)
			plot_samples(self.grid_pred, samples[task_id], alpha=0.3, fig=fig, ax=ax)
		else:
			plot_process(self.grid_pred, self.mean_pred[task_id], self.cov_pred[task_id], fig=fig, ax=ax)

		# Plot mean profile as dashed line with no CI
		plot_process(self.grid_pred, self.post_mean_grid, jnp.zeros_like(self.post_cov_grid), linestyle="--",
		             ci_alpha=0, curve_alpha=1, label="Mean profile", fig=fig, ax=ax)

		# Plot pred points
		pred_color = plt.get_cmap("tab10")(0)
		plot_points(self.padded_inputs_pred[task_id], self.padded_outputs_pred[task_id], color=pred_color, marker="o",
		            zorder=3, fig=fig, ax=ax)

		# Plot test points
		test_color = plt.get_cmap("tab10")(1)
		plot_points(self.padded_inputs_test[task_id], self.padded_outputs_test[task_id], color=test_color, marker="o",
		            zorder=3, fig=fig, ax=ax)

		return fig, ax

	def plot_mean_process(self, include_train_points=True, fig=None, ax=None):
		if fig is None and ax is None:
			fig, ax = plt.subplots(figsize=(12, 8))

		if include_train_points:
			plot_tasks(self.padded_inputs_train, self.padded_outputs_train, point_alpha=0.1, fig=fig, ax=ax)

		# Plot process
		plot_process(self.all_inputs_train, self.post_mean, self.post_cov, fig=fig, ax=ax)

		return fig, ax

	def generate_grid(self, grid_size, margin=5):
		self.grid_pred = jnp.linspace(jnp.min(self.all_inputs_train - margin, axis=0), jnp.max(self.all_inputs_train + margin, axis=0), grid_size)
		return self.grid_pred


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

		self.grid_pred = None
		self.post_means_pred = None
		self.post_covs_pred = None
		self.means_pred = None
		self.covs_pred = None

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

	def fit(self, max_iter: int = 25, converg_threshold: float = 1e-3, jitter: Array = jnp.array(1e-4), freeze_mean_hps: bool = False, freeze_task_hps: bool = False):
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
			mean_llh = 0
			if not freeze_mean_hps:
				self.mean_kernel, mean_llh = optimise_mean_kernel(self.mean_kernel, self.all_inputs_train, prior_mean_on_grid,
				                                                  self.post_means, self.post_covs, jitter=jitter)

			task_llh = 0
			if not freeze_task_hps:
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

	def optimise_pred_kernels(self, jitter: Array = jnp.array(1e-4)):
		# Optimise the task kernel for prediction
		self.task_kernel_pred, _ = optimise_task_kernel(self.task_kernel_pred, self.padded_inputs_pred, self.padded_outputs_pred,
		                                                self.mappings_pred, self.post_means, self.post_covs,
		                                                mixture_coeffs=self.mixture_train, shared_hp=self.shared_hp,
		                                                cluster_hp=self.cluster_hp, jitter=jitter)

	def predict(self, skip_retrain: bool=False, jitter: Array = jnp.array(1e-4)) -> Tuple[Array, Array]:
		if not self.shared_hp and not skip_retrain:
			self.optimise_pred_kernels()

		if self.mixture_pred is None:
			# Set mixture
			self.mixture_pred = update_mixture(
				self.task_kernel_pred, self.padded_inputs_pred, self.padded_outputs_pred, self.mappings_pred,
				self.post_means, self.post_covs,
				self.shared_hp, self.cluster_hp, jitter=jitter)

		# Merge grid and all_inputs
		full_grid = lexicographic_sort(
			jnp.unique(jnp.concatenate([self.all_inputs_train, self.all_inputs_pred, self.grid_pred]), axis=0))
		# Compute new mappings
		mappings_train_on_grid = vmap(compute_mapping, in_axes=(None, 0))(full_grid, self.padded_inputs_train)
		mappings_pred_on_grid = vmap(compute_mapping, in_axes=(None, 0))(full_grid, self.padded_inputs_pred)
		mappings_grid_on_full_grid = vmap(compute_mapping, in_axes=(None, 0))(full_grid, self.grid_pred)

		if self.cluster_hp:
			batched_hyperpost = vmap(hyperpost, in_axes=(None, None, None, None, None, None, self.task_kernel_pred.batch_in_axes, None, None, 0))
			post_means_full_grid, post_covs_full_grid = batched_hyperpost(
				self.padded_inputs_train,
				self.padded_outputs_train,
				mappings_train_on_grid,
				self.grid_pred,
				jnp.array(0.),
				self.mean_kernel,
				self.task_kernel_pred.inner_kernel,
				False,  # As we use a grid
				self.shared_hp,
				self.mixture_train)
		else:
			batched_hyperpost = vmap(hyperpost, in_axes=(None, None, None, None, None, None, None, None, None, 0))
			post_means_full_grid, post_covs_full_grid = batched_hyperpost(
				self.padded_inputs_train,
				self.padded_outputs_train,
				mappings_train_on_grid,
				self.grid_pred,
				jnp.array(0.),
				self.mean_kernel,
				self.task_kernel_pred,
				False,  # As we use a grid
				self.shared_hp,
				self.mixture_train)

		self.post_mean_grid = post_means_full_grid[:, mappings_grid_on_full_grid]
		self.post_cov_grid = vmap(lambda p: p[jnp.ix_(mappings_grid_on_full_grid, mappings_grid_on_full_grid)])(post_covs_full_grid)

		post_mean_pred = vmap(lambda m: post_means_full_grid[:, m])(mappings_pred_on_grid)
		post_cov_pred = vmap(lambda m: vmap(lambda p: p[jnp.ix_(m, m)])(post_covs_full_grid))(mappings_pred_on_grid)
		post_cov_crossed = vmap(lambda m: vmap(lambda p: p[jnp.ix_(m, mappings_grid_on_full_grid)])(post_covs_full_grid))(
			mappings_pred_on_grid)


		if self.cluster_hp:
			batched_predict = vmap(predict_single_cluster,
			                       in_axes=(None, None, None, 0, 0, 0, 0, 0, self.task_kernel_pred.batch_in_axes))
			# Compute predictions
			self.means_pred, self.covs_pred = batched_predict(
				self.padded_inputs_pred,
				self.padded_outputs_pred,
				self.grid_pred,
				jnp.swapaxes(post_mean_pred, 0, 1), jnp.swapaxes(post_cov_pred, 0, 1), jnp.swapaxes(post_cov_crossed, 0, 1),
				self.post_mean_grid, self.post_cov_grid,
				self.task_kernel_pred.inner_kernel)

		else:
			batched_predict = vmap(predict_single_cluster,
			                       in_axes=(None, None, None, 0, 0, 0, 0, 0, None))
			# Compute predictions
			self.means_pred, self.covs_pred = batched_predict(
				self.padded_inputs_pred,
				self.padded_outputs_pred,
				self.grid_pred,
				jnp.swapaxes(post_mean_pred, 0, 1), jnp.swapaxes(post_cov_pred, 0, 1), jnp.swapaxes(post_cov_crossed, 0, 1),
				self.post_mean_grid, self.post_cov_grid,
				self.task_kernel_pred)

		return self.means_pred, self.covs_pred

	def sample(self, nb_samples: int, jitter: Array = jnp.array(1e-3), key: Array = jr.PRNGKey(42)) -> Array:
		"""
		Return nb_samples random samples for each task in padded_inputs_pred and padded_outputs_pred.

		These samples are drown from either one of the mean_processes, with probability corresponding to the mixture
		of each prediction task.

		:param nb_samples: number of samples to return
		:param jitter: term for numerical stability
		:param key: PRNG key
		:return: array of shape (N, nb_samples, G), with:
			* N: the number of prediction tasks (first dimension of self.padded_inputs_pred)
			* G: the number of points in the prediction grid (last dimension of self.grid_pred)
		"""

		def sample_from_cluster(key, mean_pred, cov_pred, cluster_ids):
			subkeys = jr.split(key, len(cluster_ids))
			return vmap(lambda c, k: jr.multivariate_normal(k, mean_pred[c],
			                                                cov_pred[c] + (jitter * jnp.eye(len(cov_pred[c])))))(
				cluster_ids, subkeys)

		sample_cluster_ids = vmap(lambda t: jnp.searchsorted(t, (jnp.arange(nb_samples) + 0.5) / nb_samples))(
			jnp.cumsum(self.mixture_pred.T, axis=1))  # shape (Nb_Tasks, Nb_Samples)

		subkeys = jr.split(key, self.mixture_pred.shape[1])

		return vmap(sample_from_cluster, in_axes=(0, 0, 0, 0))(subkeys, jnp.swapaxes(self.means_pred, 0, 1),
		                                                       jnp.swapaxes(self.covs_pred, 0, 1), sample_cluster_ids)

	def plot_predictions(self, plot_as: Literal["samples", "process"] = "samples", task_id=0,
	                                colors="mixture", nb_samples: int = 100, include_train_points=True,
	                                key: Array = jr.PRNGKey(42), fig=None, ax=None):

		if colors is None:
			colors = [plt.get_cmap("tab10")(0) for _ in range(self.k)]
		elif colors == "mixture":
			colors = [plt.get_cmap("tab10")(i) for i in range(self.k)]
		else:
			colors = [colors] * self.k

		if fig is None and ax is None:
			fig, ax = plt.subplots(figsize=(12, 8))

		if include_train_points:
			plot_tasks(self.padded_inputs_train, self.padded_outputs_train, point_alpha=0.1, fig=fig, ax=ax)

		# Plot prediction
		if plot_as == "samples":
			samples = self.sample(nb_samples, key=key)  # (N, nb_samples, G)
			plot_samples(self.grid_pred, samples[task_id], alpha=0.3, fig=fig, ax=ax)
		else:
			# Plot K GPs, one per cluster, with alpha proportional to the cluster membership probability
			for k in range(self.k):
				weight = float(self.mixture_pred[k, task_id])
				if weight > 1e-6:
					plot_process(self.grid_pred, self.means_pred[k, task_id], self.covs_pred[k, task_id],
					             ci_alpha=0.3 * weight, curve_alpha=weight, color=colors[k], fig=fig, ax=ax)

		# Plot pred points
		pred_color = plt.get_cmap("tab10")(0)
		plot_points(self.padded_inputs_pred[task_id], self.padded_outputs_pred[task_id], color=pred_color, marker="o",
		            zorder=3, fig=fig, ax=ax)

		# Plot test points
		test_color = plt.get_cmap("tab10")(1)
		plot_points(self.padded_inputs_test[task_id], self.padded_outputs_test[task_id], color=test_color, marker="o",
		            zorder=3, fig=fig, ax=ax)

		return fig, ax

	def plot_mean_processes(self, include_train_points=True,
	                                   alphas: float | list[float] | Literal["mixture"] = "mixture", colors=None,
	                                   fig=None, ax=None):
		if colors is not None and len(colors) != self.k:
			raise ValueError(
				f"If colors is provided, it must have length equal to the number of clusters k={self.k}, but got length {len(colors)}.")

		if fig is None and ax is None:
			fig, ax = plt.subplots(figsize=(12, 8))

		if colors is None:
			colors = [plt.get_cmap("tab10")(i) for i in range(self.k)]

		if include_train_points:
			# Plot points with color corresponding to their cluster assignment
			clust_assign = jnp.argmax(self.mixture_train, axis=0)
			for cluster_id in range(self.k):
				cluster_points_mask = clust_assign == cluster_id
				plot_tasks(self.padded_inputs_train[cluster_points_mask],
				           self.padded_outputs_train[cluster_points_mask], point_alpha=0.1, color=colors[cluster_id],
				           fig=fig, ax=ax)

		if alphas == "mixture":
			# Plot each cluster's mean process weighted by the mixture proportions
			alphas = self.mixture_train.mean(axis=1).tolist()
		elif isinstance(alphas, float):
			alphas = [alphas] * self.k
		elif isinstance(alphas, list):
			if len(alphas) != self.k:
				raise ValueError(
					f"If alpha is a list, it must have length equal to the number of clusters k={self.k}, but got length {len(alphas)}.")
		else:
			raise ValueError(
				f"alpha must be a float, a list of floats of length k={self.k}, or 'mixture', but got {alphas}.")

		# Plot each cluster's mean process
		for mean, cov, alpha, color in zip(self.post_means, self.post_covs, alphas, colors):
			plot_process(self.all_inputs_train, mean, cov, ci_alpha=0.1, curve_alpha=alpha, color=color, fig=fig, ax=ax)

		return fig, ax

	def generate_grid(self, grid_size, margin=5):
		self.grid_pred = jnp.linspace(jnp.min(self.all_inputs_train - margin, axis=0), jnp.max(self.all_inputs_train + margin, axis=0), grid_size)
		return self.grid_pred
