import jax.numpy as jnp
from jax import jit, vmap
import pandas as pd


@jit
def _legacy_extract_id_data(_id, values, all_inputs):
	"""
	Extract data for a given ID from the values array and return a row of padded inputs, padded outputs and mask.

	:param _id:
	:param values:
	:param all_inputs:
	:return:
	"""
	padded_input = jnp.full((len(all_inputs),), jnp.nan)
	padded_output = jnp.full((len(all_inputs),), jnp.nan)
	mask = jnp.zeros((len(all_inputs),), dtype=bool)

	idx = jnp.searchsorted(all_inputs, jnp.where(values[:, 0] == _id, values[:, 1], jnp.nan))

	return padded_input.at[idx].set(values[:, 1]), padded_output.at[idx].set(values[:, 2]), mask.at[idx].set(True)


def _legacy_preprocess_db(db: pd.DataFrame):
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
	padded_inputs, padded_outputs, masks = vmap(_legacy_extract_id_data, in_axes=(0, None, None))(all_ids, db[["ID", "Input", "Output"]].values, all_inputs)

	return all_inputs, padded_inputs, padded_outputs, masks