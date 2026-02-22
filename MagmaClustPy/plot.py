"""
This script will feature functions used to plot various elements of the Magma(Clust) model, e.g:
* the posterior mean of the hyperposterior
* prediction on a new task
* samples from the posterior distribution
* ...
"""
from typing import Optional
from random import choice

# Third party
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from matplotlib import pyplot as plt


def plot_prediction(grid, post_mean, padded_inputs_train, padded_outputs_train, padded_inputs_pred, padded_outputs_pred, pred_mean, pred_cov, use_samples=True, num_samples=25, rand_key=jr.PRNGKey(0), jitter=1e-8):
	# TODO: legacy API, remove
	# Plot
	plt.figure(figsize=(12, 8))

	# Plot the hyperposterior mean
	plt.plot(grid, post_mean, label='post mean', linestyle="--", color="black")

	# plot the training profiles
	for i, (task_inputs, task_outputs) in enumerate(zip(padded_inputs_train, padded_outputs_train)):
		plt.plot(task_inputs, task_outputs, linestyle="None", marker='.', label=f'Train task {i}', alpha=0.5)

	# Plot either samples or confidence interval
	if use_samples:
		# FIXME: the samples are "wobbly". Maybe this is linked to the jitter?
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


# Utility plotting functions
def plot_points(inputs: Array, outputs: Array, color=None, marker=".", point_alpha: float=0.5, label: Optional[str]=None, zorder: float=2, fig=None, ax=None):
	"""
	Plot points with a specific color and alpha.

	:return: completed fig and ax
	"""
	# Check parameters consistency
	# fig, ax must be both provided or none
	if (fig is None) + (ax is None) not in (0, 2):
		raise ValueError("fig and ax must be both provided or none.")
	if not (inputs.ndim == 1 or inputs.ndim == 2 and inputs.shape[-1] == 1):
		raise ValueError(f"`plot_points()` only works for 1D input points, but got shape {inputs.shape}. You should call it with the specific dimension you wish to plot.")
	if not (outputs.ndim == 1 or outputs.ndim == 2 and outputs.shape[-1] == 1):
		raise ValueError(f"`plot_points()` only works for 1D output points, but got shape {outputs.shape}. You should call it with the specific dimension you wish to plot.")
	if len(inputs) != len(outputs):
		raise ValueError(f"Every input point should have a corresponding output point, but got {len(inputs)} inputs and {len(outputs)} outputs.")

	# Squeeze inputs and outputs if needed
	if inputs.ndim == 2:
		inputs = inputs.squeeze()
	if outputs.ndim == 2:
		outputs = outputs.squeeze()

	# Create fig and ax if not provided
	created_fig = False
	if fig is None and ax is None:
		fig, ax = plt.subplots(figsize=(12, 8))
		created_fig = True

	# If color is not provided, chose one in cmap's tab10
	if color is None:
		choice([plt.get_cmap("tab10")(i) for i in range(10)])

	# Plot points
	ax.scatter(inputs, outputs, color=color, alpha=point_alpha, marker=marker, label=label, zorder=zorder)

	if created_fig and label is not None:
		ax.legend()

	return fig, ax

def plot_tasks(inputs: Array, outputs: Array, color=None, point_alpha=0.5, task_label: bool=False, points_label: Optional[str]=None, zorder: float=2, fig=None, ax=None):
	"""
	Plots tasks as points. Point in the same task share the same color.
	This function is a wrapper around plot_points, calling it for every task.

	:return:
	"""
	# Check parameters consistency
	# fig, ax must be both provided or none
	if (fig is None) + (ax is None) not in (0, 2):
		raise ValueError("fig and ax must be both provided or none.")
	if not (inputs.ndim == 2 or inputs.ndim == 3 and inputs.shape[-1] == 1):
		raise ValueError(f"`plot_tasks()` only works for 1D input points in batched sequences, but got shape {inputs.shape}. You should call it with the specific dimension you wish to plot.")
	if not (outputs.ndim == 2 or outputs.ndim == 3 and outputs.shape[-1] == 1):
		raise ValueError(f"`plot_tasks()` only works for 1D output points in batched sequences, but got shape {outputs.shape}. You should call it with the specific dimension you wish to plot.")
	if len(inputs) != len(outputs):
		raise ValueError(f"Every input point should have a corresponding output point, but got {len(inputs)} inputs and {len(outputs)} outputs.")

	# Squeeze inputs and outputs if needed
	if inputs.ndim == 3:
		inputs = inputs[:, :, 0]
	if outputs.ndim == 3:
		outputs = outputs[:, :, 0]

	# Create fig and ax if not provided
	if fig is None and ax is None:
		fig, ax = plt.subplots(figsize=(12, 8))

	# Call `plot_points` for each task
	for i, (task_inputs, task_outputs) in enumerate(zip(inputs, outputs)):
		fig, ax = plot_points(task_inputs, task_outputs, color=color, point_alpha=point_alpha, label=f"Task {i}" if task_label else points_label, zorder=zorder, fig=fig, ax=ax)

	return fig, ax


def plot_process(grid: Array, mean: Array, cov: Array,
                 color=None, linestyle="-", linewidth=2, curve_alpha: float=1, ci_alpha: float=0.3, label: Optional[str]=None, zorder: float=2,
                 fig=None, ax=None):
	"""
	Plot a gaussian process as a function with confidence intervals, with a specific color and alpha.
	This function can be used to plot mean-processes as well as predictions, for example.

	:return:
	"""
	# Check parameters consistency
	# fig, ax must be both provided or none
	if (fig is None) + (ax is None) not in (0, 2):
		raise ValueError("fig and ax must be both provided or none.")
	if not (grid.ndim == 1 or grid.ndim == 2 and grid.shape[-1] == 1):
		raise ValueError(f"`plot_process()` only works for 1D grid points, but got shape {grid.shape}. You should call it with the specific dimension you wish to plot.")
	if not (mean.ndim == 1 or mean.ndim == 2 and mean.shape[-1] == 1):
		raise ValueError(f"`plot_process()` only works for scalar-valued processes, but got shape {mean.shape}. You should call it with the specific dimension you wish to plot.")
	if not (cov.ndim == 2):
		raise ValueError(f"`plot_process()` only works for scalar-valued covariances, but got shape {cov.shape}. You should call it with the specific dimension you wish to plot.")
	if len(grid) != len(mean) or len(grid) != len(cov):
		raise ValueError(f"Every grid point should have a corresponding mean process point and covariance row, but got {len(grid)} grid, {len(mean)} mean and {len(cov)} covariance.")

	# Squeeze grid and mean if needed
	if grid.ndim == 2:
		grid = grid[:, 0]
	if mean.ndim == 2:
		mean = mean[:, 0]

	# Create fig and ax if not provided
	created_fig = False
	if fig is None and ax is None:
		fig, ax = plt.subplots(figsize=(12, 8))
		created_fig = True

	# If color is not provided, chose one in cmap's tab10
	if color is None:
		choice([plt.get_cmap("tab10")(i) for i in range(10)])

	# Plot CI
	std = jnp.sqrt(jnp.diag(cov))
	ax.fill_between(grid, mean - 2*std, mean + 2*std,
							color=color, alpha=ci_alpha, label=label, zorder=zorder)

	# Plot process
	ax.plot(grid, mean, color=color, alpha=curve_alpha, linewidth=linewidth, linestyle=linestyle, label=label, zorder=zorder)

	if created_fig and label is not None:
		ax.legend()

	return fig, ax

def plot_samples(grid: Array, samples: Array,
                 color=None, linestyle="-", linewidth=2, alpha: float=1, label: Optional[str]=None, zorder: float=2,
                 fig=None, ax=None):
	"""
	Plot samples of a gaussian process as functions, with a specific color and alpha.

	:return:
	"""
	# Check parameters consistency
	# fig, ax must be both provided or none
	if (fig is None) + (ax is None) not in (0, 2):
		raise ValueError("fig and ax must be both provided or none.")
	if not (grid.ndim == 1 or grid.ndim == 2 and grid.shape[-1] == 1):
		raise ValueError(f"`plot_samples()` only works for 1D grid points, but got shape {grid.shape}. You should call it with the specific dimension you wish to plot.")
	if not (samples.ndim == 2 or samples.ndim == 3 and samples.shape[-1] == 1):
		raise ValueError(f"`plot_samples()` only works for scalar-valued processes, but got shape {samples.shape}. You should call it with the specific dimension you wish to plot.")
	if len(grid) != len(samples[0]):
		raise ValueError(f"Every sample should have a value for each grid point, but got {samples.shape} sample shape and {grid.shape} grid shape.")

	# Squeeze grid and mean if needed
	if grid.ndim == 2:
		grid = grid[:, 0]
	if samples.ndim == 3:
		samples = samples[:, :, 0]

	# Create fig and ax if not provided
	created_fig = False
	if fig is None and ax is None:
		fig, ax = plt.subplots(figsize=(12, 8))
		created_fig = True

	for sample in samples:
		# If color is not provided, chose one in cmap's tab10
		if color is None:
			choice([plt.get_cmap("tab10")(i) for i in range(10)])

		# Plot process
		ax.plot(grid, sample, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=label, zorder=zorder)

	if created_fig and label is not None:
		ax.legend()

	return fig, ax