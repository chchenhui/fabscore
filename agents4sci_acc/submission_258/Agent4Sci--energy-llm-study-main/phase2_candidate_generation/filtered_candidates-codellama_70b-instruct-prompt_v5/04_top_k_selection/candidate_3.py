def solve(numbers, k):
    if numbers and k > 0:
        sorted_list = sorted(numbers)
        largest_k_numbers = sorted_list[-k:]
        return list(reversed(largest_k_numbers))
    else:
        return []