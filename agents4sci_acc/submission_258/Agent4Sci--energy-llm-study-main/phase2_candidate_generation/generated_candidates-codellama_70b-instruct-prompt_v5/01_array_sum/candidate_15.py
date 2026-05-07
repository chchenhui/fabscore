def solve(numbers):
    if len(numbers) == 0:
        return 0
    else:
        for num in numbers:
            if not isinstance(num, (int, float)):
                raise TypeError("Invalid input. All elements must be of type int or float.")
        return sum(numbers)

# Test the function