def solve(nums):
    # Check if the list is empty
    if not nums:
        return 0

    # Initialize sum to 0
    total = 0
    
    for x in nums:
        if not (isinstance(x, int) or isinstance(x, float)):
            raise TypeError("The list contains non-numeric values.")
        
        total += x
    
    return total