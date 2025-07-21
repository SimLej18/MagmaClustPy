# Standard library imports
import os
os.environ['JAX_ENABLE_X64'] = "True"

# Local imports
from MagmaClustPy.hyperpost import hyperpost
from MagmaClustPy.linalg import cho_factor, cho_solve, map_to_full_matrix_batch, map_to_full_array_batch

# JAX imports
import jax
import jax.numpy as jnp

# Other imports
import pandas as pd
from typing import Tuple, Optional


def predict(dataset_pred: pd.DataFrame, mean_kern, task_kern, padded_inputs_pred: jnp.ndarray, padded_outputs_pred: jnp.ndarray,
            indexed_mappings_pred: jnp.ndarray, all_inputs_pred: jnp.ndarray, grid: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:

    # TO DO : Implement the documentation for predict(), and the core of the function. To do so, all you need
    #         is in the prediction.ipynb file, in the section "Custom implementation(s)".
    #         Some arguments may be missing, so feel free to add them if needed.
    #         For now, the prediction code in prediction.ipynb file do not perform multiple predictions in parallel.
    #         Feel free to add a beautiful vmap to perform it!
    
    return None, None