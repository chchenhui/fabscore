def solve(numbers):
    try:
        result = sum(numbers)
    except TypeError:
        raise TypeError("The input list contains non-numeric elements")
    return result