import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import pandas as pd


def generate_dummy_db(M: int, MIN_N: int, MAX_N: int, grid: jnp.array, key: jnp.array):
	# We fill DB with random sequences
	data = []
	for m in range(M):
		key, subkey = jax.random.split(key)
		n_points = jax.random.randint(subkey, (), MIN_N, MAX_N)
		inputs = jax.random.choice(subkey, grid, (n_points,), replace=False)
		for n, i in zip(range(n_points), inputs):
			key, subkey1, subkey2 = jax.random.split(key, 3)
			data.append({
				"Task_ID": m,
				"Input": i.item(),
				"Output": jax.random.uniform(subkey2, (), jnp.float32, -5, 5).item()
			})
	return pd.DataFrame(data)


@jit
def extract_id_data(_id, values, all_inputs, to_fill):
	"""
	Extract data for a given ID from the values array and return a row of padded inputs, padded outputs and index_mappings.

	:param _id:
	:param id_index:
	:param values:
	:param all_inputs:
	:return:
	"""
	inputs_i = jnp.where(values[:,0] == _id, values[:,1], jnp.nan)
	outputs_i = jnp.where(values[:,0] == _id, values[:,2], jnp.nan)
	mappings_i = jnp.searchsorted(all_inputs, inputs_i)

	# Compute index among the whole dataset
	idx_i = jnp.where(jnp.isnan(inputs_i), to_fill.shape[0] + 1, jnp.cumsum(~jnp.isnan(inputs_i)) - 1)

	# Create padded inputs and outputs
	padded_input = jnp.full(to_fill.shape[0], jnp.nan).at[idx_i].set(inputs_i)
	padded_output = jnp.full(to_fill.shape[0], jnp.nan).at[idx_i].set(outputs_i)
	index_mappings = jnp.full(to_fill.shape[0], all_inputs.shape[0] + 1).at[idx_i].set(mappings_i).astype(int)

	return padded_input, padded_output, index_mappings


def preprocess_db(db: pd.DataFrame):
	"""

	:param db: the db to process, with columns "ID", "Input" and "Output", in that order
	:return: a tuple of (all_inputs, padded_inputs, padded_outputs, masks)
	   - all_inputs: a matrix of shape (P, ) with all distinct inputs
	   - padded_inputs: a matrix of shape (M, MAX_N) where M is the number of sequences and MAX_N is the max number of points among all sequences. Missing inputs for each sequence are represented as NaNs.
	   - padded_outputs: a matrix of shape (M, MAX_N) with corresponding output for each input and NaNs for missing inputs
	   - index_mappings: a matrix of shape (M, MAX_N) with indices of the inputs in the all_inputs array. Missing inputs for each sequence are represented as -1.
	"""
	# Get all distinct inputs
	db_sorted = db.sort_values(['Task_ID', 'Input'])
	all_ids = jnp.array(db_sorted["Task_ID"].unique())
	all_inputs = jnp.sort(jnp.array(db_sorted["Input"].unique()))
	MAX_N = db_sorted.groupby("Task_ID")["Input"].count().max()  # Maximum number of points in a sequence
	to_fill = jnp.full((MAX_N), jnp.nan)  # Placeholder for padded inputs and outputs

	# Initialise padded inputs, padded outputs and masks
	padded_inputs, padded_outputs, index_mappings = vmap(extract_id_data, in_axes=(0, None, None, None))(all_ids,
																											 db_sorted[
																												 ["Task_ID",
																												  "Input",
																												  "Output"]].values,
																											 all_inputs,
																											 to_fill)

	return all_inputs, padded_inputs, padded_outputs, index_mappings
