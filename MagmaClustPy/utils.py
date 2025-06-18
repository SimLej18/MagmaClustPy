import jax.numpy as jnp
from jax import random, jit, vmap
import pandas as pd


def generate_dummy_db(M: int, MIN_N: int, MAX_N: int, grid: jnp.array, key: jnp.array):
	# We fill DB with random sequences
	data = []
	for m in range(M):
		key, subkey = random.split(key)
		n_points = random.randint(subkey, (), MIN_N, MAX_N)
		for n in range(n_points):
			key, subkey1, subkey2 = random.split(key, 3)
			data.append({
				"ID": m,
				"Input": random.choice(subkey1, grid, (1,))[0].item(),
				"Output": random.uniform(subkey2, (), jnp.float64, -5, 5).item()
			})
	return pd.DataFrame(data)


@jit
def extract_id_data(_id, values, all_inputs):
	"""
	Extract data for a given ID from the values array and return a row of padded inputs, padded outputs and mask.

	:param _id:
	:param id_index:
	:param values:
	:param all_inputs:
	:return:
	"""
	padded_input = jnp.full((len(all_inputs),), jnp.nan)
	padded_output = jnp.full((len(all_inputs),), jnp.nan)
	mask = jnp.zeros((len(all_inputs),), dtype=bool)

	idx = jnp.searchsorted(all_inputs, jnp.where(values[:, 0] == _id, values[:, 1], jnp.nan))

	return padded_input.at[idx].set(values[:, 1]), padded_output.at[idx].set(values[:, 2]), mask.at[idx].set(True)


def preprocess_db(db: pd.DataFrame):
	"""

	:param db: the db to process, with columns "ID", "Input" and "Output", in that order
	:return: a tuple of (all_inputs, padded_inputs, padded_outputs, masks)
	   - all_inputs: a matrix of shape (P, ) with all distinct inputs
	   - padded_inputs: a matrix of shape (M, P) where M is the number of sequences and P is the number of distinct
	   inputs. Missing inputs for each sequence are represented as NaNs.
	   - padded_outputs: a matrix of shape (M, P) with corresponding output for each input and NaNs for missing inputs
	   - masks: a matrix of shape (M, P) with 1 where the input is valid and 0 where it is padded
	"""
	# Get all distinct inputs
	all_ids = jnp.array(db["ID"].unique())
	all_inputs = jnp.sort(jnp.array(db["Input"].unique()))

	# Initialise padded inputs, padded outputs and masks
	padded_inputs, padded_outputs, masks = vmap(extract_id_data, in_axes=(0, None, None))(all_ids, db[["ID", "Input", "Output"]].values, all_inputs)

	return all_inputs, padded_inputs, padded_outputs, masks