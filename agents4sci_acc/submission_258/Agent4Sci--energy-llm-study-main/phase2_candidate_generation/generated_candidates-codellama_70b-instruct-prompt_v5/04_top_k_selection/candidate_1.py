def solve(numbers, k):
    if len(numbers) == 0 or k <= 0:
        return []
    numbers = sorted(numbers, reverse=True)
    return numbers[:k]