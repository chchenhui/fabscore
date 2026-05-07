def solve(nums):
    if len(nums) == 0:
        return []
    try:
        prefix_sum = 0
        output = []
        for num in nums:
            prefix_sum += num
            output.append(prefix_sum)
        return output
    except TypeError as e:
        raise TypeError("Invalid input list!")