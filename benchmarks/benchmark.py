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

# Other imports
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Local imports
from Kernax import SEMagmaKernel, DiagKernel, ExpKernel
from MagmaClustPy.hyperpost import hyperpost
from MagmaClustPy.hp_optimisation import optimise_hyperparameters
from MagmaClustPy.utils import preprocess_db


def run_train(dataset: str, shared_input: bool, shared_hp: bool, max_iter: int = 25, converg_threshold: float = 1e-3,
              jitter: jnp.array = jnp.array(1e-4)):
	"""
	Run the training pipeline with the specified parameters.

	:param dataset: Name of the dataset to use (e.g., 'small', 'medium', 'large', 'huge').
	:param shared_input: Whether all tasks share the same input grid.
	:param shared_hp: Whether to use shared hyperparameters across tasks.
	:param max_iter: Maximum number of iterations for the training loop.
	:param converg_threshold: Convergence threshold for the training loop.
	:param jitter: jitter term for numerical stability in the covariance matrices.
	"""
	# Check if cuda is available
	logging.info(f"Jax launched using {jax.default_backend()} backend.")

	## Start timer
	start = time.time()

	## Data import
	dataset_file = os.path.join("../datasets", f"{dataset}_{'shared_input' if shared_input else 'distinct_input'}_{'shared_hp' if shared_hp else 'distinct_hp'}.csv")
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

	## Training
	# Priors
	prior_mean = jnp.array(0)
	mean_kernel = SEMagmaKernel(length_scale=jnp.array(0.9), variance=jnp.array(1.5))

	if shared_hp:
		task_kernel = SEMagmaKernel(length_scale=jnp.array(.3), variance=jnp.array(1.)) + DiagKernel(ExpKernel(jnp.array(2.5)))
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
		logging.info(f"Iteration {i:4}\tLlhs: {prev_mean_llh:12.4f}, {prev_task_llh:12.4f}\tConv. Ratio: {conv_ratio:.5f}\t\n\tMean: {mean_kernel}\t\n\tTask: {task_kernel}")
		# e-step: compute hyper-posterior
		post_mean, post_cov = hyperpost(padded_inputs_train, padded_outputs_train, mappings_train, all_inputs_train, prior_mean, mean_kernel, task_kernel)

		# m-step: update hyperparameters
		mean_kernel, task_kernel, mean_llh, task_llh = optimise_hyperparameters(mean_kernel, task_kernel, padded_inputs_train, padded_outputs_train, mappings_train, all_inputs_train, prior_mean, post_mean, post_cov, jitter=jitter, verbose=VERBOSE)

		# Check for NaN values and stop early
		#if jnp.isnan(mean_llh) or jnp.isnan(task_llh):
		#	logging.error(f"NaN detected at iteration {i}. Stopping training.")
		#	break

		# Check convergence
		if i > 0:
			conv_ratio = jnp.abs((prev_mean_llh + prev_task_llh) - (mean_llh + task_llh)) / jnp.abs(prev_mean_llh + prev_task_llh)
			if conv_ratio < converg_threshold:
				logging.info(f"Convergence reached after {i+1} iterations.\tLlhs: {mean_llh:12.4f}, {task_llh:12.4f}\n\tMean: {mean_kernel}\n\tTask: {task_kernel}")
				break

		if i == max_iter - 1:
			logging.warning(f"Maximum number of iterations reached.\nLast modif: {jnp.abs(prev_mean_llh - mean_llh).item()} & {jnp.abs(prev_task_llh - task_llh).item()}")

		prev_mean_llh = mean_llh
		prev_task_llh = task_llh

	## Prediction
	#TODO

	## End timer
	end = time.time()
	logging.info(f"Magma finished in {end - start}s")


if __name__ == "__main__":
	# Command-line argument parsing
	parser = argparse.ArgumentParser(description='Run MagmaClustPy benchmarks')
	parser.add_argument('--dataset', type=str, default='small', help='Dataset size: small, medium, large, or huge')
	parser.add_argument('--shared_input', type=str, default='true', help='Use shared input: true or false')
	parser.add_argument('--shared_hp', type=str, default='true', help='Use shared hyperparameters: true or false')
	
	args = parser.parse_args()
	
	dataset = args.dataset
	shared_input = args.shared_input.lower() == 'true'
	shared_hp = args.shared_hp.lower() == 'true'

	grids = {
		"small": jnp.arange(-10, 10, 0.5),
		"medium": jnp.arange(-100, 100, 0.5),
		"large": jnp.arange(-500, 500, 0.5),
		"custom": jnp.arange(-20, 20, 0.5)
	}
	grid = grids[dataset] if dataset in grids else grids["custom"]

	# Default hyper-parameters
	MAX_ITER = 25
	CONVERG_THRESHOLD = 1e-3
	jitter = jnp.array(1e-4)

	run_train(dataset, shared_input, shared_hp, max_iter=MAX_ITER, converg_threshold=CONVERG_THRESHOLD, jitter=jitter)
