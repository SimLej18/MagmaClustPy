import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.stats.multivariate_normal import logpdf

from MagmaClustPy.linalg import extract_from_full_array, extract_from_full_matrix, solve_right_cholesky


@jit
def magma_neg_likelihood_on_cov(covar, outputs, mean, mean_process_cov, mapping, nugget=jnp.array(1e-10)):
	nugget_matrix = jnp.eye(outputs.shape[0]) * nugget

	eyed_covar = jnp.where(jnp.isnan(covar), jnp.eye(covar.shape[0]), covar)
	zeroed_outputs = jnp.nan_to_num(outputs)
	if mapping is not None:
		zeroed_mean = jnp.nan_to_num(extract_from_full_array(mean, outputs, mapping))
		eyed_mean_cov = jnp.where(jnp.isnan(covar), jnp.eye(covar.shape[0]), extract_from_full_matrix(mean_process_cov, outputs, mapping))
	else:
		zeroed_mean = jnp.nan_to_num(mean)
		eyed_mean_cov = jnp.where(jnp.isnan(covar), jnp.eye(covar.shape[0]), mean_process_cov)


	# Compute log-likelihood
	multiv_neg_log_lik = -logpdf(zeroed_outputs, zeroed_mean, eyed_covar + nugget_matrix)

	# Compute correction term
	correction = 0.5 * jnp.trace(solve_right_cholesky(eyed_covar, eyed_mean_cov, nugget=nugget))

	# Compute padding corrections
	# The logpdf is computed as:
	# -0.5 * (N * log(2 * pi) + log(det(cov)) + (outputs - mean).T @ inv(cov) @ (outputs - mean))
	# det(cov) and the Mahalanobis distance are not affected by our padding
	# We only have to correct for the -0.5 * N * log(2 * pi) term, as N is bigger with padding
	nll_pad_correction = 0.5 * jnp.log(2 * jnp.pi) * jnp.sum(jnp.isnan(outputs))

	# We also need to correct the correction term, as padding adds 1s to the diagonal and hence 1 to the trace
	corr_pad_correction = 0.5 * jnp.sum(jnp.isnan(outputs))

	return (multiv_neg_log_lik - nll_pad_correction) + (correction - corr_pad_correction)


@jit
def magma_neg_likelihood(kernel, inputs, outputs: jnp.array, mean: jnp.array, mean_process_cov: jnp.array, mappings: jnp.array, nugget=jnp.array(1e-10)):
	"""
	Computes the MAGMA log-likelihood.

	:param kernel: the kernel containing HPs to optimise. This kernel is used to compute the covariance (matrix `S`)
	:param inputs: inputs on which to compute the covariance matrix (shape (N, ))
	:param mask: boolean masks indicating which inputs and outputs to consider (shape (N, ))  #TODO: fix
	:param outputs: the observed values (shape (N, ))
	:param mean: the mean over the inputs (scalar or vector of shape (N, ))
	:param mean_process_cov: the hypper-posterior mean process covariance (matrix K^t)
	:param nugget: the nugget, for numerical stability

	:return: the negative log-likelihood (scalar)
	"""
	covar = kernel(inputs)

	# check if we need to vmap
	if inputs.ndim == 1:
		return magma_neg_likelihood_on_cov(covar, outputs, mean, mean_process_cov, mappings, nugget)
	elif inputs.ndim == 2:
		return vmap(magma_neg_likelihood_on_cov, in_axes=(0, 0, None, None, 0, None))(covar, outputs, mean, mean_process_cov, mappings, nugget)
	else:
		raise ValueError("inputs must be either 1D or 2D")