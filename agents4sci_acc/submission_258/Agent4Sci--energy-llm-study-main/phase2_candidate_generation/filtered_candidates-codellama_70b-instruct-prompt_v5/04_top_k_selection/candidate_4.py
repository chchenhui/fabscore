def solve(numbers, k):
    # Sort the numbers in descending order
    sorted_numbers = sorted(numbers, reverse=True)

    if not numbers or k <= 0:
        return []
    elif k >= len(numbers):
        return sorted_numbers
    else:
        return sorted_numbers[:k]