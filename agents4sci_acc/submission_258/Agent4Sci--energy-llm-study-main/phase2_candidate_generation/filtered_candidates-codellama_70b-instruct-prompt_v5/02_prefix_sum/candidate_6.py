def solve(numbers):
    if not numbers: return []
    for i in range(len(numbers)):
        if not isinstance(numbers[i], int): raise TypeError("Invalid input")
    prefix_sum = [0] * (len(numbers))
    for i in range(len(prefix_sum)):
        if i == 0:
            prefix_sum[i] = numbers[i]
        else:
            prefix_sum[i] = prefix_sum[i - 1] + numbers[i]
    return prefix_sum

# Test the function with a list of integers.
input_numbers = [2, 3, 4]
print(solve(input_numbers))