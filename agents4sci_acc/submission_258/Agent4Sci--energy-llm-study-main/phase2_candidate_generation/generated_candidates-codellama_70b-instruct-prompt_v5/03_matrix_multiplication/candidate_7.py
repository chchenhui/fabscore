def solve(matrix1, matrix2):
    # Check if both inputs are valid matrices represented by 2D lists
    if not isinstance(matrix1, list) or not all(isinstance(row, list) for row in matrix1):
        raise TypeError("Matrix1 must be a 2D list of numbers")
    if not isinstance(matrix2, list) or not all(isinstance(row, list) for row in matrix2):
        raise TypeError("Matrix2 must be a 2D list of numbers")
    
    # Check the dimensions match for multiplication
    num_cols1 = len(matrix1[0])
    num_rows2 = len(matrix2)
    if num_cols1 != num_rows2:
        raise ValueError("Dimensions do not match for matrix multiplication")
    
    # Check that all elements in both matrices are numbers (integers or floats)
    valid_elements = set([int, float])
    if not all(isinstance(element, tuple(valid_elements)) for row in matrix1 for element in row):
        raise TypeError("Matrix1 elements must be integers or floats")
    if not all(isinstance(element, tuple(valid_elements)) for row in matrix2 for element in row):
        raise TypeError("Matrix2 elements must be integers or floats")
    
    # Perform the matrix multiplication and return the result
    num_rows1 = len(matrix1)
    num_cols2 = len(matrix2[0])
    
    result_matrix = [[sum(a * b for a, b in zip(row1, col2)) for col2 in zip(*matrix2)] for row1 in matrix1]
    
    return result_matrix