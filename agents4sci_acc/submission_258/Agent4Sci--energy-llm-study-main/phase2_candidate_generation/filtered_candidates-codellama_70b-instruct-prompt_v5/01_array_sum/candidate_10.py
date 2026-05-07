def solve(numbers):
    
    # Check if the list is empty and return 0 in that case
    if not numbers:
        return 0

    for number in numbers:
        # Raise TypeError if any element is not a number (either int or float)
        if not isinstance(number, (int, float)):
            raise TypeError("Invalid input. List contains non-numeric values.")

    # If all elements are numbers, calculate and return the sum of numbers in the list
    return sum(numbers)