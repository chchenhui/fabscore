def solve(data):
    result = {}
    if data:
        for element in data:
            if element in result.keys():
                result[element] += 1
            else:
                result[element] = 1
    return result