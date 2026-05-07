# Define the matrix-multiplication function
def solve(A, B):
    # Check that elements of both matrices are numbers
    if not all(isinstance(x, (int, float)) for row in A for x in row) or \
            not all(isinstance(x, (int, float)) for row in B for x in row):
        raise TypeError("Matrices contain non-numeric elements.")
    # Check if dimensions match for multiplication
    if len(A[0]) != len(B):
        raise ValueError("Dimensions do not match for matrix multiplication.")
    # Perform matrix multiplication
    result = [[sum(a * b for a, b in zip(row_A, col_B))
               for col_B in zip(*B)] for row_A in A]
    return result