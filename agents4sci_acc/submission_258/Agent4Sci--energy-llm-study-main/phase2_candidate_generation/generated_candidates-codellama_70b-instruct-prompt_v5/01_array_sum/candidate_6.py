def solve(numbers):
    if not numbers: # check for an empty list
        return 0
    
    total = 0.0   # initialize total to be a float type, so that it can accept both int and float values in the list
    for num in numbers:
        if isinstance(num, (int, float)):  # check whether each element is an integer or float
            total += num
        else:
            raise TypeError("List contains non-numeric elements")
    
    return total