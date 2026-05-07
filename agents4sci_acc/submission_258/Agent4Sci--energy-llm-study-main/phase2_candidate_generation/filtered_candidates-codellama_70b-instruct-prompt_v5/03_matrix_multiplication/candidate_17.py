def solve(matrix1, matrix2):
    # Validate if elements are numbers and dimensions match for multiplication
    validate_inputs(matrix1, matrix2)
    
    num_rows = len(matrix1)
    num_cols = len(matrix2[0])
    
    result_matrix = [[0] * num_cols for _ in range(num_rows)]
    
    # Perform the multiplication using nested loops
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result_matrix[i][j] += matrix1[i][k] * matrix2[k][j]
    
    return result_matrix