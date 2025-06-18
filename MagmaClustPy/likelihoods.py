import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve
from jax.scipy.stats.multivariate_normal import logpdf


@jit
def solve_right_cholesky(A, B, nugget=jnp.array(1e-10)):
	""" Solves for X in X @ A = B """
	# For X @ A = B, we can transpose both sides: A.T @ X.T = B.T
	# As A and B are symmetric, this simplifies to A @ X.T = B
	# Then solve for X.T and transpose the result
	nugget_matrix = jnp.eye(A.shape[0]) * nugget
	return cho_solve(cho_factor(A + nugget_matrix), B).T


@jit
def magma_neg_likelihood_on_cov(covar, outputs, mean, mean_process_cov, mask=None, nugget=jnp.array(1e-10)):
	nugget_matrix = jnp.eye(outputs.shape[0]) * nugget

	if mask is not None:
		# Mask the covariance matrix and outputs
		mask_2D = mask[:, None] & mask[None, :]
		covar = jnp.where(mask_2D, covar, jnp.eye(outputs.shape[0]))
		outputs = jnp.where(mask, outputs, 0)
		mean = jnp.where(mask, mean, 0)
		mean_process_cov = jnp.where(mask_2D, mean_process_cov, jnp.eye(outputs.shape[0]))


	# Compute log-likelihood
	multiv_neg_log_lik = -logpdf(outputs, mean, covar + nugget_matrix)

	# Compute correction term
	correction = 0.5 * jnp.trace(solve_right_cholesky(covar, mean_process_cov, nugget=nugget))

	if mask is not None:
		# Correct log-likelihood for padding
		# The logpdf is computed as:
		# -0.5 * (N * log(2 * pi) + log(det(cov)) + (outputs - mean).T @ inv(cov) @ (outputs - mean))
		# det(cov) and the Mahalanobis distance are not affected by our padding
		# We only have to correct for the -0.5 * N * log(2 * pi) term, as N is bigger with padding
		nll_pad_correction = 0.5 * jnp.log(2 * jnp.pi) * jnp.sum(~mask, axis=0)

		# We also need to correct the correction term, as padding adds 1s to the diagonal and hence 1 to the trace
		corr_pad_correction = 0.5 * jnp.sum(~mask, axis=0)
	else:
		nll_pad_correction = 0
		corr_pad_correction = 0

	return (multiv_neg_log_lik - nll_pad_correction) + (correction - corr_pad_correction)


@jit
def magma_neg_likelihood(kernel, inputs, outputs: jnp.array, mean: jnp.array, mean_process_cov: jnp.array, mask=None,
                         nugget=jnp.array(1e-10)):
	"""
	Computes the MAGMA log-likelihood.

	:param kernel: the kernel containing HPs to optimise. This kernel is used to compute the covariance (matrix `S`)
	:param inputs: inputs on which to compute the covariance matrix (shape (N, ))
	:param mask: boolean masks indicating which inputs and outputs to consider (shape (N, ))
	:param outputs: the observed values (shape (N, ))
	:param mean: the mean over the inputs (scalar or vector of shape (N, ))
	:param mean_process_cov: the hypper-posterior mean process covariance (matrix K^t)
	:param nugget: the nugget, for numerical stability

	:return: the negative log-likelihood (scalar)
	"""
	covar = kernel(inputs)

	# check if we need to vmap
	if inputs.ndim == 1:
		return magma_neg_likelihood_on_cov(covar, outputs, mean, mean_process_cov, mask, nugget)
	elif inputs.ndim == 2:
		return vmap(magma_neg_likelihood_on_cov, in_axes=(0, 0, None, None, 0, None))(covar, outputs, mean,
		                                                                              mean_process_cov, mask, nugget)
	else:
		raise ValueError("inputs must be either 1D or 2D")