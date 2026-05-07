def solve(numbers):
    if len(numbers) == 0:
        return []

    for num in numbers:
        if not isinstance(num, int):
            raise TypeError("All elements must be integers.")

    prefix_sums = [numbers[0]]
    for i in range(1, len(numbers)):
        prefix_sums.append(prefix_sums[-1] + numbers[i])

    return prefix_sums