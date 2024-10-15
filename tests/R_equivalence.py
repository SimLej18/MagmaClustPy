from rpy2.robjects import r

# Load the R script containing the sum_function
r.source("r_functions.R")

# Get the R function from the global R environment
r_sum_function = r['sum_function']


# Python equivalent function
def sum_function(a, b):
	return a + b


def test_sum_function_equivalence():
	# Test parameters
	a = 3
	b = 5

	# Call the Python function
	python_result = sum_function(a, b)

	# Call the R function
	r_result = r_sum_function(a, b)[0]  # r_result is an R vector, so extract the first element

	# Assert that both results are equal
	assert python_result == r_result
