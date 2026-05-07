def solve(A, B):
    # Check if input matrices are valid
    m1 = len(A)
    n1 = len(A[0])
    m2 = len(B)
    n2 = len(B[0])

    # Raise TypeError if elements are not numbers
    for i in range(m1):
        for j in range(n1):
            if not isinstance(A[i][j], (int, float)):
                raise TypeError("Elements of matrix A must be integers or floats")
    for i in range(m2):
        for j in range(n2):
            if not isinstance(B[i][j], (int, float)):
                raise TypeError("Elements of matrix B must be integers or floats")

    # Check if dimensions match for multiplication
    if n1 != m2:
        raise ValueError(f"Dimensions do not match for matrix multiplication. Matrix A is {m1} x {n1}, but matrix B is {m2} x {n2}")

    # Perform matrix multiplication using dot product
    C = [[sum(x * y for x, y in zip(row_A, col_B)) for col_B in zip(*B)] for row_A in A]

    return C