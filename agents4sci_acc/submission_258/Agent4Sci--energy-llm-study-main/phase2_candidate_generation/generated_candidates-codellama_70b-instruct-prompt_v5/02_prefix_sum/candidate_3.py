def solve(input_list):
    for num in input_list:
        if not isinstance(num, int):
            raise TypeError("All elements must be integers.")

    prefix_sums = []
    current_sum = 0

    for num in input_list:
        current_sum += num
        prefix_sums.append(current_sum)

    return prefix_sums