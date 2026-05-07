def solve(numbers, k):
    if len(numbers) == 0 or k <= 0:
        return []
    numbers.sort()
    if k > len(numbers):
        return numbers[::-1]
    else:
        return numbers[-k:][::-1]