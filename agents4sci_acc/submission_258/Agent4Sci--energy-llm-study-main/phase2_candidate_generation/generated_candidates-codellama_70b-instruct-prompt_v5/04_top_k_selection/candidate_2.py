import heapq
def solve(numbers, k):
    if len(numbers) == 0 or k <= 0:
        return []
    else:
        result = heapq.nlargest(k, numbers)
        return sorted(result, reverse=True)