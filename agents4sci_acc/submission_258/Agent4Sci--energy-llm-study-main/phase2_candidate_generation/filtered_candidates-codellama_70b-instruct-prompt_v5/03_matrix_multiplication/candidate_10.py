def solve(A, B):
    # Check if matrices have valid dimensions
    rows_a = len(A)
    cols_a = len(A[0])
    rows_b = len(B)
    cols_b = len(B[0])

    # Check if inner matrix dimensions are equal (requirement for matrix multiplication)
    if cols_a != rows_b:
        raise ValueError("Matrices cannot be multiplied due to incompatible dimensions.")

    # Initialize empty output matrix with zeros
    C = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    # Perform element-wise multiplication and addition (algorithm for matrix product)
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                C[i][j] += A[i][k] * B[k][j]

    return C