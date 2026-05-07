def solve(numbers, k):
    if not numbers or k <= 0:
        return []
    sorted_numbers = sorted(numbers, reverse=True)
    if k > len(sorted_numbers):
        return sorted_numbers
    else:
        return sorted_numbers[:k]