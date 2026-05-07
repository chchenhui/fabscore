def solve(nums):
    """
    Function that calculates and returns the prefix sums of a list of integers.
    
    Parameters:
        nums (list): A list of integers
        
    Returns:
        prefix_sums (list): A list containing the prefix sums.
    """
    if not nums:
        return [] # Return an empty list if the input is empty

    for idx in range(len(nums)):
        try:  # Try to convert each element to int, raise TypeError if it's not possible
            nums[idx] = int(nums[idx])
        except ValueError:
            raise TypeError("All elements must be integers.")
        
    prefix_sums = []
    
    for i in range(len(nums)): # Iterate over the length of the input list
        if i == 0:
            prefix_sums.append(nums[i]) # Add the first element to the prefix_sums list directly
        else:
            prefix_sums.append(prefix_sums[-1] + nums[i]) # Calculate and append the sum of previous sums with current value
    return prefix_sums