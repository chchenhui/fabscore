def solve(numbers):
    result = {}
    for number in numbers:
        if number in result:
            result[number] += 1
        else:
            result[number] = 1
    return result