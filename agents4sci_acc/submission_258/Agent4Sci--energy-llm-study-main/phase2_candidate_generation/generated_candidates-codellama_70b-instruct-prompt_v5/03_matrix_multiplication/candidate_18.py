def solve(matrix1, matrix2):
    m1 = len(matrix1)
    n1 = len(matrix1[0])
    m2 = len(matrix2)
    n2 = len(matrix2[0])
    if n1 != m2:
        raise ValueError("Dimensions do not match for matrix multiplication")
    
    result = [[0] * n2 for _ in range(m1)]
    for i in range(m1):
        for j in range(n2):
            for k in range(m2):
                if not isinstance(matrix1[i][k], (int, float)) or not isinstance(matrix2[k][j], (int, float)):
                    raise TypeError("Inputs contain invalid elements")
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result