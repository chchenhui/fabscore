def solve(numbers, k):
    numbers = sorted(numbers, reverse=True)  # Sort list in descending order
    if not numbers or k <= 0:                # If list empty or k <= 0 return []
        return []
    if len(numbers) < k:                     # If k is larger than list length, return the whole list
        return numbers
    else:
        return numbers[:k]  # Return first k largest elements in descending order