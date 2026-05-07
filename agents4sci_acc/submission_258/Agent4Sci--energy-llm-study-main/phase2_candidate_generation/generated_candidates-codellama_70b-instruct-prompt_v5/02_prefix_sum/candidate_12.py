def solve(list_of_numbers):
    if not all(isinstance(num, int) for num in list_of_numbers):
        raise TypeError("All elements must be integers.")
    prefix_sums = []
    current_sum = 0
    for number in list_of_numbers:
        current_sum += number
        prefix_sums.append(current_sum)
    return prefix_sums