def solve(numbers):
    if len(numbers) == 0:
        return {}
    else:
        frequency = {}
        for number in numbers:
            if number in frequency:
                frequency[number] += 1
            else:
                frequency[number] = 1
        return frequency