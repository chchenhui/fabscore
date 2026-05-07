def solve(numbers):
    if not numbers:
        return []
    
    # Check if all elements are integers and raise TypeError otherwise
    for num in numbers:
        if type(num) != int:
            raise TypeError("Input list must contain only integers")
            
    prefix_sums = [numbers[0]]  # Initialize with the first number
    for i in range(1, len(numbers)):
        prefix_sums.append(prefix_sums[-1] + numbers[i])
    
    return prefix_sums