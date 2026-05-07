import heapq

def solve(numbers, k):
    if not numbers or k <= 0:
        return []
    
    # Use max-heap to get the largest k elements in O(nlogk) time.
    result = heapq.nlargest(k, numbers)

    return sorted(result, reverse=True)