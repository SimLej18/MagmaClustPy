"""
This module provides linear algebra functions for MagmaClustPy.
"""

import jax.numpy as jnp
import jax.scipy as jsc
from jax import jit, vmap
from jax.lax import cond, while_loop


# --- Linear algebra functions ---

@jit
def cho_factor(cov, initial_jitter=1e-10, max_jitter=1.0):
	"""
	Wrapper around jax.scipy.linalg.cho_factor to compute the Cholesky factorisation of a covariance matrix.
	It always returns the upper factorisation, as we use the upper version of Cholesky in the whole codebase.
	It automatically gets the smallest jitter that makes the covariance matrix PSD

	:param cov: Covariance matrix to factorise
	:param initial_jitter: Initial jitter value to start with (default is 1e-10)
	:param max_jitter: Maximum jitter value to try (default is 1.0)

	:return: Cholesky upper factorisation of the covariance matrix. If it still contains NaNs, it means the matrix is still not PSD even with the maximum jitter.
	"""

	def condition(jitter):
		# Go on while the factorisation fails AND the jitter is under the limit
		cov_jittered = cov + jitter * jnp.eye(len(cov))
		chol_result = jsc.linalg.cho_factor(cov_jittered)[0]
		has_nan = jnp.any(jnp.isnan(chol_result))
		return jnp.logical_and(has_nan, jitter < max_jitter)

	def body(jitter):
		print("Current jitter:", jitter)
		new_jitter = jitter * 10
		return new_jitter

	# Initialisation
	final_jitter = while_loop(condition, body, initial_jitter)

	# Return the factorisation with the final jitter
	return jsc.linalg.cho_factor(cov + final_jitter * jnp.eye(len(cov)))[0]


@jit
def cho_solve(factor, B):
	"""
	Wrapper around jax.scipy.linalg.cho_solve to solve the linear system A @ X = B, as we always use the upper version
	of Cholesky factorisation in the whole codebase.

	:param factor: The Cholesky factorisation of the covariance matrix A (output of cho_factor).
	:param B: The right-hand side matrix or vector to solve for.

	:return: The solution X such that A @ X = B.
	"""
	return cho_solve((factor, False), B)


@jit
def solve_right_cholesky(A, B):
	""" Solves for X in X @ A = B """
	# For X @ A = B, we can transpose both sides: A.T @ X.T = B.T
	# As A and B are symmetric, this simplifies to A @ X.T = B
	# Then solve for X.T and transpose the result
	return cho_solve(cho_factor(A), B).T


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


@jit
def searchsorted_2D(vector, matrix):
	"""
	Search along axis 1 for a vector in a matrix. If found, return the index of the vector.
	If not found, return len(matrix).

	For this function to work, the vectors in the matrix must be sorted lexicographically.
	ex:
	[[1, 1, 0],
	 [1, 2, 1],
	 [1, 2, 2],
	 [2, 1, 3],
	 [2, 2, 1]]

	:param vector: the vector to search for
	:param matrix: the matrix to search in
	:return: the index of the vector in the matrix, or len(matrix) if not found
	"""

	@jit
	def compare_vectors(v1, v2):
		"""Compare two vectors lexicographically. Returns -1 if v1 < v2, 0 if equal, 1 if v1 > v2"""
		diff = v1 - v2
		# Find first non-zero element
		nonzero_mask = diff != 0
		# If all elements are zero, vectors are equal
		first_nonzero_idx = jnp.argmax(nonzero_mask)

		return cond(
			jnp.any(nonzero_mask),
			lambda: jnp.array(jnp.sign(diff[first_nonzero_idx]), dtype=jnp.int32),
			lambda: jnp.array(0, dtype=jnp.int32)
		)

	@jit
	def search_condition(state):
		start, end, found = state
		return (start < end) & (~found)

	@jit
	def search_step(state):
		start, end, found = state
		mid = (start + end) // 2

		comparison = compare_vectors(vector, matrix[mid])

		# If vectors are equal, we found it
		new_found = comparison == 0
		new_start = cond(comparison < 0, lambda: start, lambda: mid + 1)
		new_end = cond(comparison < 0, lambda: mid, lambda: end)

		# If found, return the index in start position
		final_start = cond(new_found, lambda: mid, lambda: new_start)

		return final_start, new_end, new_found

	# Initial state: (start, end, found)
	initial_state = (0, len(matrix), False)
	final_start, final_end, found = while_loop(search_condition, search_step, initial_state)

	# Return the found index or len(matrix) if not found
	return cond(found, lambda: final_start, lambda: len(matrix))


searchsorted_2D_vectorized = jit(vmap(searchsorted_2D, in_axes=(0, None)))
