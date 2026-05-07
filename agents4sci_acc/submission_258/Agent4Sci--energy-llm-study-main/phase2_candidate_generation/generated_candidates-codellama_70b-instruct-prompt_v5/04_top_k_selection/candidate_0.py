def solve(numbers, k):
    numbers.sort(reverse=True)
    if not numbers or k <= 0:
        return []
    else:
        return numbers[:k]