import numpy as np
def solve(mat1, mat2):
    if len(mat1) != len(mat2[0]):
        raise ValueError("Invalid dimensions for matrix multiplication.")
    else: 
        try:
            result = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*mat2)] for X_row in mat1]
        except TypeError:
            raise TypeError("Elements must be numbers.")
    return np.array(result)