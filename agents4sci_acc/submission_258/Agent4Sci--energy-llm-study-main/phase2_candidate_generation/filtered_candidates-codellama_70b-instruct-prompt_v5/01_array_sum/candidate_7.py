def solve(numbers):
    if len(numbers) == 0:
        return 0
    
    for number in numbers:
        if not (isinstance(number, int) or isinstance(number, float)):
            raise TypeError("List contains non-numeric elements")
            
    return sum(numbers)