# Toy functions to test rpy2 and pytest
library(tibble)

sum_function <- function(a, b) {
    return(a + b)
}

modify_df <- function(df) {
  df$new_column <- df$a + df$b  # Add a new column that's the sum of 'a' and 'b'
  return(df)  # or `return(as_tibble(df))` for a tibble, works either way
}