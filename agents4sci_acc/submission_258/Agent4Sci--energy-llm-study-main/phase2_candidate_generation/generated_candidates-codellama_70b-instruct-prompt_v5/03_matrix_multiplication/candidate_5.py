def solve(A, B):
    """Function to multiply two matrices in Python without using any external libraries."""
    
    # Check that the number of columns in A matches the number of rows in B
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions do not match for multiplication.")
        
    # Initialize an empty result matrix with shape (rows_A, cols_B) 
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    
    # Calculate the matrix product using nested loops:
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
                
    return result