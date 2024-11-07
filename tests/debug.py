import logging

import pandas as pd

from MagmaClustPy import lin_alg_backend as lab
from MagmaClustPy.simulate_data import simu_db
from MagmaClustPy.training import train_magma
from MagmaClustPy.kernels import SquaredExponentialKernel, SquaredExponentialMagmaKernel

logging.basicConfig(level=logging.INFO)
#lab.change_backend("mlx")

# db = simu_db(k=3)

db = pd.DataFrame({
    'ID': [1, 1, 1, 1, 2, 2, 2, 2],
    'Input': [0.40, 4.45, 7.60, 8.30, 3.50, 5.10, 8.85, 9.35],
    'Output': [59.81620, 67.13694, 78.32495, 81.83590, 62.04943, 67.31932, 85.94063, 86.76426]
})

kern_0 = SquaredExponentialMagmaKernel(length_scale=0.3, variance=1.0, noise=0)
kern_0_magma = SquaredExponentialMagmaKernel(length_scale=0.3, variance=1.0, noise=0)
kern_i = SquaredExponentialMagmaKernel(length_scale=0.3, variance=1.0, noise=-2.5)
kern_i_magma = SquaredExponentialMagmaKernel(length_scale=0.3, variance=1.0, noise=-2.5)

train_magma(db, kern_0=kern_0_magma, kern_i=kern_i_magma)
