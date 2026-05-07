def solve(matrix1, matrix2):
    # Check if number of columns in matrix1 is equal to number of rows in matrix2
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Dimensions do not match for multiplication")
    
    # Initialize the result matrix with zeros
    result = [[0] * len(matrix2[0]) for _ in range(len(matrix1))]
    
    # Perform element-wise multiplication and summing up
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    
    return result