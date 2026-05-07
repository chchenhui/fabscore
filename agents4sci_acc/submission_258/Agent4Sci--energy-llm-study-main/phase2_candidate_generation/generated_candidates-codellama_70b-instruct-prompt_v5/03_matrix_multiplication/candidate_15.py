# Function to find out matrix multiplication
def solve(A, B):
    # Checking if the elements of matrices are numbers
    for i in A:
        for j in range(len(i)):
            if not isinstance(i[j], int) and not isinstance(i[j], float):
                raise TypeError("Matrix contains non-numeric elements")
    
    for i in B:
        for j in range(len(i)):
            if not isinstance(i[j], int) and not isinstance(i[j], float):
                raise TypeError("Matrix contains non-numeric elements")

    # Checking if the dimensions are valid for multiplication
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])
    
    if colsA != rowsB:
        raise ValueError("Dimensions do not match for matrix multiplication")
    
    # Multiplying the two matrices (matrix product)
    C = [[sum(a*b for a,b in zip(A_row,B_col)) for B_col in zip(*B)] for A_row in A]
    return C