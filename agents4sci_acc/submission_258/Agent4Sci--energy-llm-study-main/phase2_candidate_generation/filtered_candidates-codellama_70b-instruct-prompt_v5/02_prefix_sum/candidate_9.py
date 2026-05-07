def solve(numbers):
    if len(numbers) == 0: return []

    # Check for invalid elements. Raise a TypeError if found.
    for num in numbers:
        if not isinstance(num, int):
            raise TypeError("Invalid input element")

    prefix_sums = [numbers[0]]
    
    for i in range(1, len(numbers)):
        sum = prefix_sums[-1] + numbers[i]
        prefix_sums.append(sum)

    return prefix_sums