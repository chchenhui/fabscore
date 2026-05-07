def solve(nums):
    prefix_sum = []
    if len(nums) == 0:
        return prefix_sum
    for num in nums:
        if not isinstance(num, int):
            raise TypeError("Expected all elements to be integers")
        if len(prefix_sum) > 0:
            current_total = num + prefix_sum[-1]
        else:
            current_total = num
        prefix_sum.append(current_total)
    return prefix_sum