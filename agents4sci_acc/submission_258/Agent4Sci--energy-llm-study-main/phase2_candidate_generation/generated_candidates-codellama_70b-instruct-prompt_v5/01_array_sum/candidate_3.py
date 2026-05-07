def solve(numbers):
    # Check if input list is empty
    if len(numbers) == 0:
        return 0
    
    sum = 0
    for number in numbers:
        # Raise TypeError exception if a value is not a number
        if not isinstance(number, (int, float)):
            raise TypeError('Input list contains non-numeric values')
        
        sum += number
    
    return sum