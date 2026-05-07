def solve(input_list):
    if len(input_list) == 0:
        return []
    for element in input_list:
        if not isinstance(element, int):
            raise TypeError("Input list must contain only integers")
    prefix_sums = [0]
    current_sum = 0
    for number in input_list:
        current_sum += number
        prefix_sums.append(current_sum)
    return prefix_sums[1:]