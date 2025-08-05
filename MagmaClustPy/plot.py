"""
This script will feature functions used to plot various elements of the Magma(Clust) model, e.g:
* the posterior mean of the hyperposterior
* prediction on a new task
* samples from the posterior distribution
* ...
"""
# Third party
import jax.numpy as jnp
import jax.random as jr

from matplotlib import pyplot as plt


def plot_prediction(grid, post_mean, padded_inputs_train, padded_outputs_train, padded_inputs_pred, padded_outputs_pred, pred_mean, pred_cov, use_samples=True, num_samples=25, rand_key=jr.PRNGKey(0), jitter=1e-8):
	# Plot
	plt.figure(figsize=(12, 8))

	# Plot the hyperposterior mean
	plt.plot(grid, post_mean, label='post mean', linestyle="--", color="black")

	# plot the training profiles
	for i, (task_inputs, task_outputs) in enumerate(zip(padded_inputs_train, padded_outputs_train)):
		plt.plot(task_inputs, task_outputs, linestyle="None", marker='.', label=f'Train task {i}', alpha=0.5)

	# Plot either samples or confidence interval
	if use_samples:
		samples = jr.multivariate_normal(rand_key, pred_mean, pred_cov + (jitter * jnp.eye(len(pred_cov))), shape=(num_samples,))

		# Plot the samples
		for i in range(num_samples):
			plt.plot(grid, samples[i], linestyle='-', color='pink', alpha=0.3)
	else:  # Plot the prediction confidence interval
		plt.fill_between(grid.flatten(),
				 pred_mean - 1.98 * jnp.sqrt(jnp.diag(pred_cov)),
				 pred_mean + 1.98 * jnp.sqrt(jnp.diag(pred_cov)),
				 color='pink', alpha=0.5, label='Predicted confidence interval')

	# plot prediction mean
	plt.plot(grid, pred_mean, label='Predicted mean', color='purple', alpha=0.5)

	# plot the points of the task we want to predict
	plt.plot(padded_inputs_pred, padded_outputs_pred, marker='o', linestyle="None", label='Pred data', color='black')

	# Labels and legend
	plt.xlabel("Input")
	plt.ylabel("Output")
	#TODO set x and y ticks dynamically
	plt.xticks([0, 5, 10])
	plt.yticks([0, 40, 80])
	# plt.legend()
	# plt.grid()
	plt.show()