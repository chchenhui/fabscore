def solve(integers):
    if not all(isinstance(i, int) for i in integers):
        raise TypeError("All elements must be of type 'int'")
    prefix_sums = []
    current_sum = 0
    for num in integers:
        current_sum += num
        prefix_sums.append(current_sum)
    return prefix_sums