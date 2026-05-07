def solve(matrix1, matrix2):
    # Check the number of columns in the first matrix is equal to the number of rows in the second matrix
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Dimensions do not match for multiplication")

    # Verify that all elements are numbers
    for row in matrix1:
        for element in row:
            if type(element) not in (int, float):
                raise TypeError("Matrix contains non-numeric values.")

    for row in matrix2:
        for element in row:
            if type(element) not in (int, float):
                raise TypeError("Matrix contains non-numeric values.")

    # Initialize the resulting matrix with zeros.
    result = [[0] * len(matrix2[0]) for _ in range(len(matrix1))]

    # Perform matrix multiplication using a nested loop
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result