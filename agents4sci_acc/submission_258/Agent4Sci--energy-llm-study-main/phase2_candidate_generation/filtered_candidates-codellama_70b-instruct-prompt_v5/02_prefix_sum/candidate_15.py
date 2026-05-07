def solve(nums):
    prefix_sum = []

    for num in nums:
        if type(num) != int:
            raise TypeError("Input must be a list of integers")
        if len(prefix_sum) == 0:
            prefix_sum.append(num)
        else:
            previous_sum = prefix_sum[-1]
            current_sum = previous_sum + num
            prefix_sum.append(current_sum)
    return prefix_sum