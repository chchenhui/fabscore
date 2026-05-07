def solve(A, B):
    """Return the matrix product of two matrices A and B."""
    
    # Check if elements are numbers
    for row in A + B:
        for element in row:
            if not isinstance(element, (int, float)):
                raise TypeError("Matrix elements must be integers or floats.")
    
    # Check dimensions
    if len(A[0]) != len(B):
        raise ValueError("Matrices have inconsistent dimensions for multiplication.")
    
    C = []  # Output matrix
    
    for i in range(len(A)):
        row_C = [0]*len(B[0])  # Initialize the row of C with zeros
        
        for j in range(len(B[0])):
            for k in range(len(B)):
                row_C[j] += A[i][k] * B[k][j]
    
        C.append(row_C)  # Add the new row to the result matrix
        
    return C