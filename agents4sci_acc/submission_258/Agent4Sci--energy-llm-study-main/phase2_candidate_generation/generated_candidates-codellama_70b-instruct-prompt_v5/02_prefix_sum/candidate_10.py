def solve(numbers):
    prefix_sums = []
    for i in numbers:
        if not isinstance(i, int):
            raise TypeError("Element must be an integer")
        else:
            prefix_sums.append((prefix_sums or [0])[-1] + i)
    return prefix_sums