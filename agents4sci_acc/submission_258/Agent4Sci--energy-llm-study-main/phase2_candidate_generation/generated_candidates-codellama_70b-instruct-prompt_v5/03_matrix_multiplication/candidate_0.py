def solve(a, b):
    if len(a) != len(b[0]):  # Checking if the dimensions match for matrix multiplication
        raise ValueError("Matrices dimensions do not match")

    result = [[sum(i*j for i, j in zip(row_a, col_b)) for col_b in zip(*b)] for row_a in a]  # Matrix multiplication using list comprehension

    return result