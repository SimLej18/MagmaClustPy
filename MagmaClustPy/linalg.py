"""
This module provides linear algebra functions for MagmaClustPy.
"""

import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve


# --- Linear algebra functions ---

@jit
def solve_right_cholesky(A, B, nugget=jnp.array(1e-10)):
	""" Solves for X in X @ A = B """
	# For X @ A = B, we can transpose both sides: A.T @ X.T = B.T
	# As A and B are symmetric, this simplifies to A @ X.T = B
	# Then solve for X.T and transpose the result
	nugget_matrix = jnp.eye(A.shape[0]) * nugget
	return cho_solve(cho_factor(A + nugget_matrix), B).T


# --- Mapping functions ---

@jit
def map_to_full_matrix(dense_cov, all_inputs, mapping):
	return jnp.full((len(all_inputs), len(all_inputs)), jnp.nan).at[jnp.ix_(mapping, mapping)].set(dense_cov)


@jit
def map_to_full_matrix_batch(dense_covs, all_inputs, mappings):
	return vmap(map_to_full_matrix, in_axes=(0, None, 0))(dense_covs, all_inputs, mappings)


@jit
def map_to_full_array(dense_array, all_inputs, mapping):
	return jnp.full((len(all_inputs)), jnp.nan).at[mapping].set(dense_array)


@jit
def map_to_full_array_batch(dense_arrays, all_inputs, mappings):
	return vmap(map_to_full_array, in_axes=(0, None, 0))(dense_arrays, all_inputs, mappings)


@jit
def extract_from_full_array(full_array, like, mapping):
	return jnp.where(mapping < len(full_array), full_array[mapping], jnp.nan)

@jit
def extract_from_full_array_batch(full_arrays, like, mappings):
	return vmap(extract_from_full_array, in_axes=(0, None, 0))(full_arrays, like, mappings)


@jit
def extract_from_full_matrix(full_matrix, like, mapping):
	mg = jnp.meshgrid(mapping, mapping)
	return jnp.where((mg[0] < len(full_matrix)) & (mg[1] < len(full_matrix)), full_matrix[mg[0], mg[1]], jnp.nan)

@jit
def extract_from_full_matrix_batch(full_matrices, like, mappings):
	return vmap(extract_from_full_matrix, in_axes=(0, None, 0))(full_matrices, like, mappings)
