def solve(matrix1, matrix2):
    if not (isinstance(matrix1, list) and isinstance(matrix2, list)):
        raise TypeError("Invalid input: lists expected")
    for row in matrix1 + matrix2:
        if not isinstance(row, list):
            raise TypeError("Invalid input: 2D lists expected")
    m = len(matrix1)
    n = len(matrix2[0])
    p = len(matrix1[0])
    if len(matrix2) != p:
        raise ValueError("Incompatible dimensions for matrix multiplication")
    result = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for k in range(p):
                if not isinstance(matrix1[i][k], (int, float)) or not isinstance(matrix2[k][j], (int, float)):
                    raise TypeError("Invalid input: numbers expected")
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result