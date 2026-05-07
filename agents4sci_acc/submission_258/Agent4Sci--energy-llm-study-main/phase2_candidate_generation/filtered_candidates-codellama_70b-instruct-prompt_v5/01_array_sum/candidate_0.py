def solve(numbers):
    if len(numbers) == 0:
        return 0
    
    for number in numbers:
        if not isinstance(number, (int, float)):
            raise TypeError("All elements must be numbers.")
            
    return sum(numbers)