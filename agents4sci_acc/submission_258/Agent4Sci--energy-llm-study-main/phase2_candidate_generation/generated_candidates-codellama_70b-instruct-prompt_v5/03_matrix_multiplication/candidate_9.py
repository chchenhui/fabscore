def solve(matrix1, matrix2):
    try:
        m1 = len(matrix1)
        n1 = len(matrix1[0])
        m2 = len(matrix2)
        n2 = len(matrix2[0])

        if n1 != m2:
            raise ValueError("Dimensions do not match for matrix multiplication.")

        result_matrix = [[0 for _ in range(n2)] for _ in range(m1)]

        for i in range(m1):
            for j in range(n2):
                for k in range(m2):
                    if isinstance(matrix1[i][k], int) or isinstance(matrix1[i][k], float) and \
                       isinstance(matrix2[k][j], int) or isinstance(matrix2[k][j], float):
                        result_matrix[i][j] += matrix1[i][k] * matrix2[k][j]
                    else:
                        raise TypeError("Matrix elements must be numbers.")

        return result_matrix

    except (TypeError, ValueError) as e:
        print(e)