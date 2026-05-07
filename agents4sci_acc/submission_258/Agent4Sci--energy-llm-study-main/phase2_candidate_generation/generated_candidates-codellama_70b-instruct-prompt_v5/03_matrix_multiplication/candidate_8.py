# Function to calculate matrix product
def solve(matrix1, matrix2):
    # Check if dimensions are valid for multiplication
    if len(matrix1[0]) == len(matrix2):
        result = [[sum(x * y for x, y in zip(row1, col))
                   for col in zip(*matrix2)] for row1 in matrix1]
        return result  # Return the product matrix
    else:
        raise ValueError("Matrix dimensions do not match.")