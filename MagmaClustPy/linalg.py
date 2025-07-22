"""
This module provides linear algebra functions for MagmaClustPy.
"""

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap
from jax.lax import cond, while_loop


# --- Linear algebra functions ---

@jit
def cho_factor(cov: jnp.ndarray, init_jitter: jnp.ndarray = jnp.array(1e-10),
               max_jitter: jnp.ndarray = jnp.array(1.0)) -> jnp.ndarray:
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

		factorisation = jsp.linalg.cho_factor(cov + new_jitter * eye)[0]

		return factorisation, new_jitter, eye

	if cov.ndim == 2:
		eye = jnp.eye(cov.shape[-1])
	elif cov.ndim == 3:
		eye = jnp.eye(cov.shape[-1])[None, :, :]
	else:
		raise ValueError(f"Invalid covariance matrix shape: {cov.shape}. Expected 2D or 3D array.")

	init_factorisation = jsp.linalg.cho_factor(cov + init_jitter * eye)[0]

	# Initialisation
	carry = (init_factorisation, init_jitter, eye)
	final_factorisation, _, _ = while_loop(condition, body, carry)

	# Return the factorisation with the final jitter
	return final_factorisation


@jit
def cho_solve(factor: jnp.ndarray, result: jnp.ndarray) -> jnp.ndarray:
	"""
	Wrapper around jax.scipy.linalg.cho_solve to solve the linear system factor @ X = result, as we always use the upper version
	of Cholesky factorisation in the whole codebase.

	:param factor: The Cholesky factorisation of the covariance matrix A (output of cho_factor).
	:param result: The right-hand side matrix or vector to solve for.

	:return: The solution X such that factor @ X = result.

	Special notes on broadcasting:
	- if factor is a matrix (shape M, M) and result is a vector (shape M,), the output will be a vector (shape M,).
	- if factor is a matrix (shape M, M) and result is a matrix (shape M, N), the output will be a matrix (shape M, N).
	- if factor is a matrix (shape M, M) and result is a batch of matrices (shape B, M, N), the output will be a batch of matrices (shape B, M, N).
	- if factor is a batch of matrices (shape B, M, M) and result is a vector (shape M,), the output will be a batch of vectors (shape B, M).
	- if factor is a batch of matrices (shape B, M, M) and result is a matrix (shape M, N), the output will be a batch of matrices (shape B, M, N).
	- if factor is a batch of matrices (shape B, M, M) and result is a batch of vectors (shape B, M), the output will be a batch of vectors (shape B, M).
	- if factor is a batch of matrices (shape B, M, M) and result is a batch of matrices (shape B, M, N), the output will be a batch of matrices (shape B, M, N).

	The broadcasting has one ambiguous case: when factor.ndim==3 and result.ndim==2. Depending on the first dimension, we have two resolutions:
	- (B, M, M) + (M, M) -> (B, M, M), aka each matrix in the batch is solved against the same matrix
	- (B, M, M) + (B, M) -> (B, M), aka each matrix in the batch is solved against its corresponding vector
	If B == M, we have no way of knowing which option to choose. By default, the first option is picked.
	If you want to avoid that, you can add a dimension at the end of the vectors batch and squeeze the result, e.g: (B, M, M) + (B, M, 1) -> (B, M, 1)
	"""
	# Handle different broadcasting scenarios
	if factor.ndim == 2:
		# factor is (M, M)
		if result.ndim == 2 and result.shape[1] == factor.shape[0] and result.shape[0] != factor.shape[0]:
			# Case: (M, M) + (B, M) -> (B, M): transpose result to (M, B), solve, then transpose back
			return jsp.linalg.cho_solve((factor, False), result.T).T
		else:
			# Standard cases: (M, M) + (M,) or (M, M) + (M, N)
			return jsp.linalg.cho_solve((factor, False), result)
	elif factor.ndim == 3:
		# factor is (B, M, M)
		if result.ndim == 1:
			# (B, M, M) + (M,) -> (B, M): broadcast result to match batch dimension
			result = jnp.broadcast_to(result, (factor.shape[0], result.shape[0]))
		elif result.ndim == 2:
			# Handle ambiguous case: (B, M, M) + (B, M) vs (B, M, M) + (M, N)
			if result.shape[0] == factor.shape[0] and result.shape[1] == factor.shape[1]:
				# Case (B, M, M) + (B, M) -> (B, M): already compatible
				pass
			elif result.shape[0] == factor.shape[1]:
				# Case (B, M, M) + (M, N) -> (B, M, N): broadcast to batch dimension
				result = jnp.broadcast_to(result, (factor.shape[0],) + result.shape)
	# For (B, M, M) + (B, M, N), no broadcasting needed

	return jsp.linalg.cho_solve((factor, False), result)


@jit
def solve_right_cholesky(A: jnp.ndarray, B: jnp.ndarray, jitter: jnp.ndarray = jnp.array(1e-10)) -> jnp.ndarray:
	""" Solves for X in X @ A = B """
	# Note: this function doesn't use an adaptative jitter, because it's used in the optimisation process.
	# As jax autodiff doesn't support while loops, we cannot use the cho_factor function with an adaptative jitter.

	# For X @ A = B, we can transpose both sides: A.T @ X.T = B.T
	# As A and B are symmetric, this simplifies to A @ X.T = B
	# Then solve for X.T and transpose the result
	jitter_matrix = jnp.eye(A.shape[0]) * jitter
	return jsp.linalg.cho_solve(jsp.linalg.cho_factor(A + jitter_matrix), B).T


# --- Mapping functions ---

@jit
def map_to_full_matrix(dense_cov: jnp.ndarray, all_inputs: jnp.ndarray, mapping: jnp.ndarray) -> jnp.ndarray:
	return jnp.full((len(all_inputs), len(all_inputs)), jnp.nan).at[jnp.ix_(mapping, mapping)].set(dense_cov)


@jit
def map_to_full_matrix_batch(dense_covs: jnp.ndarray, all_inputs: jnp.ndarray, mappings: jnp.ndarray) -> jnp.ndarray:
	return vmap(map_to_full_matrix, in_axes=(0, None, 0))(dense_covs, all_inputs, mappings)


@jit
def map_to_full_array(dense_array: jnp.ndarray, all_inputs: jnp.ndarray, mapping: jnp.ndarray) -> jnp.ndarray:
	return jnp.full((len(all_inputs)), jnp.nan).at[mapping].set(dense_array)


@jit
def map_to_full_array_batch(dense_arrays: jnp.ndarray, all_inputs: jnp.ndarray, mappings: jnp.ndarray) -> jnp.ndarray:
	return vmap(map_to_full_array, in_axes=(0, None, 0))(dense_arrays, all_inputs, mappings)


@jit
def extract_from_full_array(full_array: jnp.ndarray, like: jnp.ndarray, mapping: jnp.ndarray) -> jnp.ndarray:
	return jnp.where(mapping < len(full_array), full_array[mapping], jnp.nan)


@jit
def extract_from_full_array_batch(full_arrays: jnp.ndarray, like: jnp.ndarray, mappings: jnp.ndarray) -> jnp.ndarray:
	return vmap(extract_from_full_array, in_axes=(0, None, 0))(full_arrays, like, mappings)


@jit
def extract_from_full_matrix(full_matrix: jnp.ndarray, like: jnp.ndarray, mapping: jnp.ndarray) -> jnp.ndarray:
	mg = jnp.meshgrid(mapping, mapping)
	return jnp.where((mg[0] < len(full_matrix)) & (mg[1] < len(full_matrix)), full_matrix[mg[0], mg[1]], jnp.nan)


@jit
def extract_from_full_matrix_batch(full_matrices: jnp.ndarray, like: jnp.ndarray, mappings: jnp.ndarray) -> jnp.ndarray:
	return vmap(extract_from_full_matrix, in_axes=(0, None, 0))(full_matrices, like, mappings)


@jit
def searchsorted_2D(vector: jnp.ndarray, matrix: jnp.ndarray) -> jnp.ndarray:
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


@jit
def lexicographic_sort(arr: jnp.ndarray) -> jnp.ndarray:
	"""
	sorts a 2D array lexicographically
	:param arr: 2D array to be sorted
	:return: sorted version along the first dimension
	"""
	return arr[jnp.lexsort(arr.T[::-1])]


@jit
def compute_mapping(grid: jnp.array, element: jnp.array) -> jnp.array:
	"""
	Returns the indices of each vector/element in the grid. The grid must be sorted lexicographically.

	:param grid: a sorted grid of shape (N,) or (N, I). If it is 2D, it must be sorted lexicographically.
	:param elements: element to search for in the grid, either scalar or of shape (I,)
	:return: indices in grid where the element can be found, as a scalar.
	"""
	if grid.shape[-1] == 1:
		# We only have 1 input dimension, and we can use the fast jnp.searchsorted function
		return jnp.searchsorted(grid.squeeze(axis=-1), element.squeeze(axis=-1))
	# Multiple input dimensions requires our custom lexicographic search
	return searchsorted_2D_vectorized(element, grid)
