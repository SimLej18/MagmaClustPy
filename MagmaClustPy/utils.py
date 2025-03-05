import jax.numpy as jnp
from jax import random
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


def preprocess_db(db: pd.DataFrame):
	"""

	:param db: the db to process, with columns "ID", "Input" and "Output"
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
	padded_inputs = jnp.full((len(all_ids), len(all_inputs)), jnp.nan)
	padded_outputs = jnp.full((len(all_ids), len(all_inputs)), jnp.nan)
	masks = jnp.zeros((len(all_ids), len(all_inputs)), dtype=bool)

	# Fill padded inputs, padded outputs and masks
	for i, _id in enumerate(db["ID"].unique()):
		sub_db = db[db["ID"] == _id]
		idx = jnp.searchsorted(all_inputs, jnp.array(sub_db["Input"]))
		padded_inputs = padded_inputs.at[i, idx].set(sub_db["Input"].values)
		padded_outputs = padded_outputs.at[i, idx].set(sub_db["Output"].values)
		masks = masks.at[i, idx].set(jnp.ones(len(sub_db), dtype=bool))

	return all_inputs, padded_inputs, padded_outputs, masks
