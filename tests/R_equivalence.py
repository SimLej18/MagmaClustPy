import pandas as pd
import numpy as np
from rpy2.robjects import FloatVector, r, pandas2ri
from rpy2.robjects.packages import importr

from MagmaClustPy import simulate_data

# Activate pandas <-> R conversion
pandas2ri.activate()

# Import R packages
tibble = importr('tibble')
magmaclust = importr('MagmaClustR')

# Load the R script containing the modify_tibble function
r.source("r_functions.R")


class TestRtoPyToyExample:

	def setup_method(self):
		""" This method is called before every test. """
		# Create a sample DataFrame for testing
		self.df_python = pd.DataFrame({
			'a': [1, 2, 3],
			'b': [4, 5, 6]
		})

	def test_sum_function_equivalence(self):
		# R function
		r_sum_function = r['sum_function']

		# Python equivalent function
		def sum_function(x, y):
			return x + y

		# Test parameters
		a = 3
		b = 5

		# Call the Python function
		python_result = sum_function(a, b)

		# Call the R function
		r_result = r_sum_function(a, b)[0]  # r_result is an R vector, so extract the first element

		# Assert that both results are equal
		assert python_result == r_result

	def test_dataframe_equivalence(self):
		""" Test the equivalence of R and Python functions working on dataframes. """
		# R function
		r_modify_df = r['modify_df']

		# Python equivalent function
		def modify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
			df['new_column'] = df['a'] + df['b']
			return df

		# Call the Python function
		df_python_modified = modify_dataframe(self.df_python.copy())

		# Call the R function (converts Pandas to R dataframe automatically)
		df_r_modified = r_modify_df(self.df_python)

		# Convert the R tibble back to a Pandas DataFrame
		df_r_modified = pandas2ri.rpy2py(df_r_modified)

		# Assert that both DataFrames are equal
		pd.testing.assert_frame_equal(df_python_modified.reset_index(drop=True), df_r_modified.reset_index(drop=True))
		# Reset index because it's a 'str' in R and 'int' in Python

class TestSimuDB:

	def test_simu_indiv(self):
		# inputs
		_input = [0.05, 0.40, 0.95]
		mean = [1, -1, -0.5]

		# Py versions
		py_input = np.array(_input)
		py_mean = np.array(mean)

		# R versions
		r_input = FloatVector(_input)
		r_mean = FloatVector(mean)

		# Call functions
		py_version = simulate_data.simu_indiv_se(_id="0", _input=py_input, mean=py_mean, v=1, l=1, sigma=0)
		r_version = pandas2ri.rpy2py(magmaclust.simu_indiv_se(ID="0", input=r_input, mean=r_mean, v=1, l=1, sigma=0))

		# Dataframes have the same length
		assert len(py_version) == len(r_version)

		# Dataframes have the same columns
		assert py_version.columns.equals(r_version.columns)

	def test_simu_db(self):
		# Call functions
		py_version = simulate_data.simu_db(k=3)
		r_version = pandas2ri.rpy2py(magmaclust.simu_db(K=3))

		# Dataframes have the same length
		assert len(py_version) == len(r_version)

		# Dataframes have the same columns
		assert py_version.columns.equals(r_version.columns)
