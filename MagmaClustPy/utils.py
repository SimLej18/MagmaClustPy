from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
from jax import jit
from jax import vmap

from MagmaClustPy.linalg import compute_mapping


def generate_dummy_db(M: int, INPUTS_ID: List[str], MIN_N: int, MAX_N: int, OUTPUTS_ID: List[str],
                      GRIDS: List[jnp.ndarray], OUTPUT_RANGES: List[jnp.ndarray], drop_output_rate: float = 0.,
                      key: jnp.ndarray = jax.random.PRNGKey(42)) -> pd.DataFrame:
	"""
	Generate a dummy database with random inputs and outputs, following the expected structure for MagmaClustPy.

	:param M: Number of tasks.
	:param INPUTS_ID: List of input IDs, each representing a dimension of inputs.
	:param MIN_N: Minimum number of inputs per task.
	:param MAX_N: Maximum number of inputs per task.
	:param OUTPUTS_ID: List of output IDs, each representing a dimension of outputs.
	:param GRIDS: List of grids to pick inputs from, one for each input dimension.
	:param drop_output_rate: Probability of dropping an output value. Default is 0, meaning no outputs are dropped.
	:param key: JAX random key for reproducibility.

	:return: A pandas DataFrame with columns "ID", "Input", "Input_ID", "Output", "Output_ID".
	"""
	data = []

	for m in range(M):
		key, subkey1, subkey2 = jr.split(key, 3)

		n_points = jr.randint(subkey1, (), MIN_N, MAX_N).item()  # This task's number of points
		inputs = [jr.choice(subkey2, grid, (n_points,), replace=g != 0) for g, grid in
		          enumerate(GRIDS)]  # Randomly pick inputs from the grids
		# We set replace=False for the first grid, to ensure we have distinct inputs in at least one dimension.

		for n in range(n_points):
			for o, output_id in enumerate(OUTPUTS_ID):
				key, subkey1, subkey2 = jr.split(key, 3)

				if jr.uniform(subkey1) < drop_output_rate:
					# Drop output value with a certain probability
					continue

				output_val = jr.uniform(subkey2, (), jnp.float32, *OUTPUT_RANGES[o])

				for i, input_id in enumerate(INPUTS_ID):
					data.append({
						"Task_ID": m,
						"Input": inputs[i][n].item(),
						"Input_ID": input_id,
						"Output": output_val.item(),
						"Output_ID": output_id
					})

	return pd.DataFrame(data)


def pivot_db(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Flatten a dataframe so that every line corresponds to a single observation.
	For example, if the "Input_ID" column has a multi-index with "x", "y", and "z", the resulting DataFrame will have
	columns like "Input_x", "Input_y", and "Input_z". If the "Output_ID" column has a multi-index with "a" and "b",
	the resulting DataFrame will have columns like "Output_a" and "Output_b".

	When an output dimension is missing for a given observation, the corresponding column will be filled with NaN.

	:param df: DataFrame with columns "Task_ID", "Input_ID", "Input", "Output", "Output_ID"
	:return: DataFrame "Task_ID" and "Input_*" and "Output_*" columns for each "Input_ID" and "Output_ID"
	"""
	# Ensure the dataframe has the expected columns
	if not all(col in df.columns for col in ["Task_ID", "Input_ID", "Input", "Output", "Output_ID"]):
		raise ValueError("DataFrame must contain 'Task_ID', 'Input_ID', 'Input', 'Output', and 'Output_ID' columns.")

	# Ensure the DataFrame is sorted by "Task_ID", "Input_ID", "Input", and "Output"
	df_flat = df.sort_values(by=["Task_ID", "Output_ID", "Output", "Input_ID"]).reset_index(drop=True)

	# Flatten Inputs
	df_flat = df_flat.pivot_table(
		index=['Task_ID', 'Output', 'Output_ID'],
		columns='Input_ID',
		values='Input',
		aggfunc='first').reset_index()

	df_flat.columns = [f"Input_{col}" if col not in ["Task_ID", "Input_ID", "Output", "Output_ID"] else col for col in
	                   df_flat.columns]
	df_flat.columns.name = None

	# Flatten Outputs
	df_flat = df_flat.pivot_table(
		index=['Task_ID'] + [col for col in df_flat.columns if col.startswith("Input_")],
		columns='Output_ID',
		values='Output',
		aggfunc='first'
	).reset_index()

	df_flat.columns = [f'Output_{col}' if col not in ['Task_ID', 'Output_ID'] and not col.startswith("Input_") else col
	                   for col in df_flat.columns]

	df_flat.columns.name = None

	return df_flat


@partial(jit, static_argnames=["max_n_i"])
def extract_task_data(_id, task_ids, input_values, output_values, all_inputs, max_n_i):
	"""
	Extract data for a given task ID from the values array and return a row of padded inputs, padded outputs and index_mappings.

	:param _id: the task ID to extract data for
	:param task_ids: the array of the task id of each observation (shape: (N, 1))
	:param input_values: the input values for all tasks  (shape: (T, N, I))
	:param output_values: the output values for all tasks (shape: (T, N, O))
	:param all_inputs: the array of all distinct inputs (shape: (P, I))
	:param max_n_i: the maximum number of inputs per task (scalar)

	:return: a tuple of (padded_input, padded_output, index_mappings)
	   - padded_input: a matrix of shape (MAX_N_I, I) with inputs for the task, padded with NaNs
	   - padded_output: a matrix of shape (MAX_N_I, O) with corresponding outputs for each input, padded with NaNs
	   - index_mappings: a matrix of shape (MAX_N_I,) with indices of the inputs in the all_inputs array. Missing inputs for the task are represented as NaNs.
	"""
	inputs_i = jnp.where(task_ids == _id, input_values, jnp.nan)
	outputs_i = jnp.where(task_ids == _id, output_values, jnp.nan)
	mappings_i = compute_mapping(all_inputs, input_values)

	# Compute index among the whole dataset
	idx_i = jnp.where(jnp.isnan(inputs_i[:, 0]), max_n_i + 1, jnp.cumsum(~jnp.isnan(inputs_i[:, 0])) - 1)

	# Create padded inputs and outputs
	padded_input = jnp.full((max_n_i, inputs_i.shape[-1]), jnp.nan).at[idx_i].set(inputs_i)
	padded_output = jnp.full((max_n_i, outputs_i.shape[-1]), jnp.nan).at[idx_i].set(outputs_i)
	index_mappings = jnp.full((max_n_i,), all_inputs.shape[0] + 1).at[idx_i].set(mappings_i).astype(int)

	return padded_input, padded_output, index_mappings


def preprocess_db(db: pd.DataFrame) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
	"""
	Turn a pandas DataFrame into a JAX-friendly format, extracting inputs, outputs, and mappings.

	:param db: Pandas DataFrame with columns "Task_ID", "Input", "Input_ID", "Output", "Output_ID"
	:return: A tuple containing:
		- padded_inputs: A JAX array of shape (T, MAX_N, I) with padded inputs for each task.
		- padded_outputs: A JAX array of shape (T, MAX_N, O) with padded outputs for each task.
		- index_mappings: A JAX array of shape (T, MAX_N) with indices of the inputs in the all_inputs array.
		- all_inputs: A JAX array of all distinct inputs of shape (N, I), where N is the number of distinct inputs.
	"""
	# Pivot the database
	db_flat = pivot_db(db)

	# Get task IDs
	task_ids = jnp.array(db_flat["Task_ID"].values.astype(int))[:, None]  # Convert to column vector

	# Get inputs and outputs
	inputs = jnp.array(db_flat.filter(like="Input_").values).astype(float)
	outputs = jnp.array(db_flat.filter(like="Output_").values).astype(float)

	# Get all distinct inputs
	all_inputs = jnp.unique(
		db_flat.sort_values(by=[col for col in db_flat if col.startswith("Input_")]).filter(like="Input_").values,
		axis=0)

	# Get maximum number of inputs per task
	MAX_N = jnp.max(jnp.sum(task_ids == task_ids[0], axis=0)).item()  # Maximum number of inputs per task

	# Recover padded inputs, padded outputs and index mappings
	padded_inputs, padded_outputs, index_mappings = (vmap(extract_task_data, in_axes=(0, None, None, None, None, None))
	                                                 (jnp.unique(task_ids), task_ids, inputs,  outputs, all_inputs, MAX_N))

	# return jnp.round(padded_inputs, 6), jnp.round(padded_outputs, 6), jnp.round(index_mappings, 6), jnp.round(all_inputs, 6)
	return padded_inputs, padded_outputs, index_mappings, all_inputs


def check_db(db: pd.DataFrame):
	"""
	Makes preliminary checks on a database used for Magma(Clust) to prevent errors during runtime, providing more explicit error messages.
	:param db: the database to check
	"""
	# TODO
	pass


def split_db(db: pd.DataFrame, train_ratio: float = 0.9, pred_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	Splits a database into training, pred and test sets based on given ratios.
	:param db: pandas DataFrame with columns "Task_ID", "Input", "Input_ID", "Output", "Output_ID"
	:param train_ratio: float, ratio of IDs to use for training (default 0.9)
	:param pred_ratio: float, ratio of inputs per pred ID to use for pred (default 0.7)
	:return: tuple of pandas DataFrames (db_train, db_pred, db_test)
	"""
	# First train_ratio% of IDs are for training, last 100-train_ratio% for testing
	train_ids = db["Task_ID"].unique()[:int(train_ratio * db["Task_ID"].nunique())]
	pred_ids = db["Task_ID"].unique()[int(train_ratio * db["Task_ID"].nunique()):]

	db_train = db[db["Task_ID"].isin(train_ids)]
	db_pred = db[db["Task_ID"].isin(pred_ids)]

	# First pred_ratio% of inputs of each pred_id is for pred, last 100-pred_ratio% for test
	db_test = db_pred.groupby("Task_ID", group_keys=False).apply(
	    lambda x: x.iloc[int(pred_ratio * len(x)):]
	)
	db_pred = db_pred.groupby("Task_ID", group_keys=False).apply(
	    lambda x: x.iloc[:int(pred_ratio * len(x))]
	)

	return db_train.reset_index(drop=True), db_pred.reset_index(drop=True), db_test.reset_index(drop=True)