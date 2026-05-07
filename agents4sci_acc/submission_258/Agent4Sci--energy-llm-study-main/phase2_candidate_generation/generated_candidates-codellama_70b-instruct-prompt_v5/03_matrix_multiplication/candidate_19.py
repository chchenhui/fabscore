#!/usr/bin/env python3
import numpy as np
def solve(a, b):
    try:
        result = np.matmul(a, b) # Multiply the matrices
        return result.tolist()   # Convert to a list of lists and return
    except ValueError as e:       # If dimensions do not match for multiplication, raise a ValueError
        print("ValueError: Dimensions do not match for matrix multiplication")
    except TypeError as e:        # If elements are not numbers, raise a TypeError
        print("TypeError: Matrix elements must be numbers")
    except Exception as e:        # Catch any other exception and raise it again
        raise(e)