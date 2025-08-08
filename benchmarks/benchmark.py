# -*- coding: utf-8 -*-

# Jax configuration
USE_JIT = True
USE_X64 = False
DEBUG_NANS = False
VERBOSE = False

# Standard library imports
import os

os.environ['JAX_ENABLE_X64'] = str(USE_X64).lower()
import time
import argparse

# JAX imports
import jax

jax.config.update("jax_disable_jit", not USE_JIT)
jax.config.update("jax_debug_nans", DEBUG_NANS)
import jax.numpy as jnp
from jax import vmap

# Other imports
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Local imports
from Kernax import SEMagmaKernel, DiagKernel, ExpKernel
from MagmaClustPy.hyperpost import hyperpost
from MagmaClustPy.hp_optimisation import optimise_mean_kernel, optimise_task_kernel
from MagmaClustPy.utils import preprocess_db
from MagmaClustPy.linalg import lexicographic_sort, compute_mapping
from MagmaClustPy.prediction import predict


def run_train(dataset: str, shared_input: bool, shared_hp: bool, max_iter: int = 25, converg_threshold: float = 1e-3,
              grid_size:int = 100, grid_margin:float = 5., jitter: jnp.ndarray = jnp.array(1e-4)) -> None:
	"""
	Run the training pipeline with the specified parameters.

	:param dataset: Name of the dataset to use (e.g., 'small', 'medium', 'large', 'huge').
	:param shared_input: Whether all tasks share the same input grid.
	:param shared_hp: Whether to use shared hyperparameters across tasks.
	:param max_iter: Maximum number of iterations for the training loop.
	:param converg_threshold: Convergence threshold for the training loop.
	:param grid_size: Size of the grid for the dataset.
	:param grid_margin: Margin for the grid around the dataset.
	:param jitter: jitter term for numerical stability in the covariance matrices.
	"""
	# Check if cuda is available
	logging.info(f"Jax launched using {jax.default_backend()} backend.")

	## Start timer
	start = time.time()

	## Data import
	dataset_file = os.path.join("datasets",
	                            f"{dataset}_{'shared_input' if shared_input else 'distinct_input'}_{'shared_hp' if shared_hp else 'distinct_hp'}.csv")
	try:
		db = pd.read_csv(dataset_file)
	except FileNotFoundError:
		logging.error(f"Dataset file not found: {dataset_file}")
		return
	# db has 3 columns: ID, Input, Output
	#
	# First 90% of IDs are for training, last 10% for testing
	train_ids = db["Task_ID"].unique()[:int(0.9 * db["Task_ID"].nunique())]
	test_ids = db["Task_ID"].unique()[int(0.9 * db["Task_ID"].nunique()):]

	db_train = db[db["Task_ID"].isin(train_ids)]
	db_test = db[db["Task_ID"].isin(test_ids)]
	# N.b: data is already sort by ID and Input in the toy datasets, but in a real case scenario, we would need to sort it

	## Data preprocessing
	# We need to convert the dataframe into jax arrays
	padded_inputs_train, padded_outputs_train, mappings_train, all_inputs_train = preprocess_db(db_train)
	padded_inputs_pred, padded_outputs_pred, mappings_pred, all_inputs_pred = preprocess_db(db_test)

	loading_end = time.time()
	logging.info(f"Dataset loading and preprocessing done in {loading_end - start:.2f}s")

	## Training
	# Priors
	prior_mean = jnp.array(0)
	mean_kernel = SEMagmaKernel(length_scale=jnp.array(0.9), variance=jnp.array(1.5))

	if shared_hp:
		task_kernel = SEMagmaKernel(length_scale=jnp.array(.3), variance=jnp.array(1.)) + DiagKernel(
			ExpKernel(jnp.array(2.5)))
	else:
		length_scales = jnp.array([0.3] * padded_inputs_train.shape[0])
		variances = jnp.array([1.] * padded_inputs_train.shape[0])
		noises = jnp.array([-2.5] * padded_inputs_train.shape[0])
		task_kernel = SEMagmaKernel(length_scale=length_scales, variance=variances) + DiagKernel(ExpKernel(noises))

	# Training loop
	prev_mean_llh = jnp.inf
	prev_task_llh = jnp.inf
	conv_ratio = jnp.inf

	for i in range(max_iter):
		logging.info(
			f"Iteration {i:4}\tLlhs: {prev_mean_llh:12.4f}, {prev_task_llh:12.4f}\tConv. Ratio: {conv_ratio:.5f}\t\n\tMean: {mean_kernel}\t\n\tTask: {task_kernel}")
		# e-step: compute hyper-posterior
		post_mean, post_cov = hyperpost(padded_inputs_train, padded_outputs_train, mappings_train, all_inputs_train,
		                                prior_mean, mean_kernel, task_kernel)

		# m-step: update hyperparameters
		mean_kernel, mean_llh = optimise_mean_kernel(mean_kernel, all_inputs_train, prior_mean, post_mean, post_cov,
		                                             jitter=jitter)
		task_kernel, task_llh = optimise_task_kernel(task_kernel, padded_inputs_train, padded_outputs_train,
		                                             mappings_train, post_mean, post_cov, jitter=jitter)

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
					f"Convergence reached after {i + 1} iterations.\tLlhs: {mean_llh:12.4f}, {task_llh:12.4f}\n\tMean: {mean_kernel}\n\tTask: {task_kernel}")
				break

		if i == max_iter - 1:
			logging.warning(
				f"Maximum number of iterations reached.\nLast modif: {jnp.abs(prev_mean_llh - mean_llh).item()} & {jnp.abs(prev_task_llh - task_llh).item()}")

		prev_mean_llh = mean_llh
		prev_task_llh = task_llh

	training_end = time.time()
	logging.info(f"Training completed in {training_end - loading_end:.2f}s")

	## Prediction
	# If distinct hyperparameters are used, we need to optimise a prediction task kernel
	if not shared_hp:
		# Initialise the task kernel for prediction
		length_scales = jnp.array([0.3] * padded_inputs_pred.shape[0])
		variances = jnp.array([1.] * padded_inputs_pred.shape[0])
		noises = jnp.array([-2.5] * padded_inputs_pred.shape[0])
		task_kernel_pred = SEMagmaKernel(length_scale=length_scales, variance=variances) + DiagKernel(ExpKernel(noises))

		# Optimise the task kernel for prediction
		task_kernel_pred, _ = optimise_task_kernel(task_kernel_pred, padded_inputs_pred, padded_outputs_pred,
		                                             mappings_pred, post_mean, post_cov, jitter=jitter)
		pred_retrain_end = time.time()
		logging.info(f"Optimised task kernel for prediction in {pred_retrain_end - training_end:.2f}s")
	else:
		task_kernel_pred = task_kernel
		pred_retrain_end = time.time()

	# Define the grid for prediction
	grid = jnp.linspace(jnp.min(all_inputs_train - grid_margin, axis=0),
	                    jnp.max(all_inputs_train + grid_margin, axis=0), grid_size)

	# Merge grid and all_inputs and compute new mappings
	full_grid = lexicographic_sort(jnp.unique(jnp.concatenate([all_inputs_train, all_inputs_pred, grid]), axis=0))
	# Compute new mappings
	mappings_train_on_grid = vmap(compute_mapping, in_axes=(None, 0))(full_grid, padded_inputs_train)
	mappings_pred_on_grid = vmap(compute_mapping, in_axes=(None, 0))(full_grid, padded_inputs_pred)

	# Compute the hyper-posterior on the grid
	post_mean_grid, post_cov_grid = hyperpost(inputs=padded_inputs_train,
	                                          outputs=padded_outputs_train,
	                                          mappings=mappings_train_on_grid,
	                                          all_inputs=full_grid,
	                                          prior_mean=jnp.array(0.),
	                                          mean_kernel=mean_kernel,
	                                          task_kernel=task_kernel)

	# Compute predictions
	pred_mean, pred_cov = predict(post_mean_grid, post_cov_grid, padded_outputs_pred, mappings_pred_on_grid, full_grid,
	                              task_kernel_pred)

	prediction_end = time.time()
	logging.info(f"Prediction completed in {prediction_end - pred_retrain_end:.2f}s")

	## End timer
	full_pipeline_end = time.time()
	logging.info(f"Magma finished in {full_pipeline_end - start}s total")


if __name__ == "__main__":
	# Command-line argument parsing
	parser = argparse.ArgumentParser(description='Run MagmaClustPy benchmarks')
	parser.add_argument('--dataset', type=str, default='small', help='Dataset size: small, medium, large, or huge')
	parser.add_argument('--shared_input', type=str, default='true', help='Use shared input: true or false')
	parser.add_argument('--shared_hp', type=str, default='true', help='Use shared hyperparameters: true or false')
	parser.add_argument('--max_iter', type=int, default=25, help='Maximum number of iterations for training')
	parser.add_argument('--converg_threshold', type=float, default=1e-3, help='Convergence threshold for training')
	parser.add_argument('--grid_size', type=int, default=100, help='Size of the grid for the dataset')
	parser.add_argument('--grid_margin', type=float, default=5., help='Margin for the grid around the dataset')
	parser.add_argument('--jitter', type=float, default=1e-4, help='Jitter term for numerical stability in covariance matrices')

	args = parser.parse_args()

	dataset = args.dataset
	shared_input = args.shared_input.lower() == 'true'
	shared_hp = args.shared_hp.lower() == 'true'
	max_iter = args.max_iter
	converg_threshold = args.converg_threshold
	grid_size = args.grid_size
	grid_margin = args.grid_margin
	jitter = jnp.array(args.jitter)

	run_train(dataset, shared_input, shared_hp, max_iter, converg_threshold, grid_size, grid_margin, jitter)
