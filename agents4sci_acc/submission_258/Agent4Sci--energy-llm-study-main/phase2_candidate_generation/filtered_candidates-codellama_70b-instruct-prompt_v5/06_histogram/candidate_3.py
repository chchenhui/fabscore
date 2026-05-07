def solve(numbers):
    frequency = {}
    for item in numbers:
        if (item in frequency):
            frequency[item] += 1
        else:
            frequency[item] = 1

    return frequency