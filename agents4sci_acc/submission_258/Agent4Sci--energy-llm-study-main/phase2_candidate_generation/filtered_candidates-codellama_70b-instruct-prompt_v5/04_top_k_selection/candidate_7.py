def solve(numbers, k):
    if (len(numbers) == 0 or k <= 0):
      return []
    
    # Sort the list in descending order
    numbers = sorted(numbers, reverse=True)
  
    # Return the first k largest numbers in the list
    return numbers[:k]