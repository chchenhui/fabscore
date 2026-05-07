def solve(matrix_a, matrix_b):
    """ Perform matrix multiplication with input checking and error handling."""

    # Check if elements are numbers
    for row in matrix_a + matrix_b:
        if not all([isinstance(x, (int, float)) for x in row]):
            raise TypeError("Matrix elements must be numbers.")

    # Check dimensions match
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError(
            "Matrix A's column size does not match Matrix B's row size.")

    # Perform matrix multiplication
    result = [[sum(a * b for a, b in zip(row_a, col_b))
               for col_b in zip(*matrix_b)] for row_a in matrix_a]

    return result