def solve(A, B):
    try:
        # Check if elements are numbers
        for matrix in (A, B):
            for row in matrix:
                for element in row:
                    assert isinstance(element, (int, float))
        
        # Validate dimensions
        if len(A[0]) != len(B):
            raise ValueError("Dimensions do not match")
    
        # Matrix multiplication
        result = [[sum(a*b for a, b in zip(row_A, col_B)) for col_B in zip(*B)] for row_A in A]
        
        return result
    except ValueError as e:
        raise e
    except TypeError as e:
        raise TypeError("Matrix elements must be numbers")