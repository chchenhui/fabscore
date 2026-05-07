def solve(input_list):
    if len(input_list) == 0:
        return []

    for i in input_list:
        if not isinstance(i, int):
            raise TypeError("All elements must be integers.")

    prefix_sums = [input_list[0]]
    for i in range(1, len(input_list)):
        prefix_sums.append(prefix_sums[-1] + input_list[i])

    return prefix_sums