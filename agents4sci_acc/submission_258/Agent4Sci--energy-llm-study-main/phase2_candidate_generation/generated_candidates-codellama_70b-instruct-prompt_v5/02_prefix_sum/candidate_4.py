def solve(numbers):
    if len(numbers) == 0:
        return []
    
    # Check if all elements are integers
    for number in numbers:
        if not isinstance(number, int):
            raise TypeError("All elements must be integers")

    prefix_sums = [numbers[0]]
    current_sum = numbers[0]
    for i in range(1, len(numbers)):
        current_sum += numbers[i]
        prefix_sums.append(current_sum)

    return prefix_sums