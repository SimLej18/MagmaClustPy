"""
This module provides linear algebra functions for MagmaClustPy.
"""

import jax.numpy as jnp
import jax.scipy as jsc
from jax import jit, vmap
from jax.lax import cond, while_loop
from jax.scipy.stats.multivariate_normal import logpdf as jax_logpdf


# --- Linear algebra functions ---

@jit
def cho_factor(cov, init_jitter=jnp.array(1e-10), max_jitter=jnp.array(1.0)):
	"""
	Wrapper around jax.scipy.linalg.cho_factor to compute the Cholesky factorisation of a covariance matrix.
	It always returns the upper factorisation, as we use the upper version of Cholesky in the whole codebase.
	It automatically gets the smallest jitter that makes the covariance matrix PSD

	:param cov: Covariance matrix to factorise
	:param init_jitter: Initial jitter value to start with (default is 1e-10)
	:param max_jitter: Maximum jitter value to try (default is 1.0)

	:return: Cholesky upper factorisation of the covariance matrix. If it still contains NaNs, it means the matrix is still not PSD even with the maximum jitter.
	"""

	def condition(carry):
		factorisation, jitter, _ = carry

		has_nan = jnp.any(jnp.isnan(factorisation))
		return jnp.logical_and(has_nan, jitter < max_jitter)

	def body(carry):
		factorisation, jitter, eye = carry
		new_jitter = jitter * 10

		factorisation = jsc.linalg.cho_factor(cov + new_jitter * eye)[0]

		return factorisation, new_jitter, eye

	if cov.ndim == 2:
		eye = jnp.eye(cov.shape[-1])
	elif cov.ndim == 3:
		eye = jnp.eye(cov.shape[-1])[None, :, :]
	else:
		raise ValueError(f"Invalid covariance matrix shape: {cov.shape}. Expected 2D or 3D array.")

	init_factorisation = jsc.linalg.cho_factor(cov + init_jitter * eye)[0]

	# Initialisation
	carry = (init_factorisation, init_jitter, eye)
	final_factorisation, _, _ = while_loop(condition, body, carry)

	# Return the factorisation with the final jitter
	return final_factorisation


@jit
def cho_solve(factor, B):
	"""
	Wrapper around jax.scipy.linalg.cho_solve to solve the linear system A @ X = B, as we always use the upper version
	of Cholesky factorisation in the whole codebase.

	:param factor: The Cholesky factorisation of the covariance matrix A (output of cho_factor).
	:param B: The right-hand side matrix or vector to solve for.

	:return: The solution X such that A @ X = B.
	"""
	return jsc.linalg.cho_solve((factor, False), B)


@jit
def solve_right_cholesky(A, B, jitter=jnp.array(1e-10)):
	""" Solves for X in X @ A = B """
	# Note: this function doesn't use an adaptative jitter, because it's used in the optimisation process.
	# As jax autodiff doesn't support while loops, we cannot use the cho_factor function with an adaptative jitter.

	# For X @ A = B, we can transpose both sides: A.T @ X.T = B.T
	# As A and B are symmetric, this simplifies to A @ X.T = B
	# Then solve for X.T and transpose the result
	jitter_matrix = jnp.eye(A.shape[0]) * jitter
	return jsc.linalg.cho_solve(jsc.linalg.cho_factor(A + jitter_matrix), B).T


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
