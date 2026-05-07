def solve(input_list):
    if len(input_list) == 0:
        return {}
    else:
        frequency = {}
        for element in input_list:
            if element in frequency:
                frequency[element] += 1
            else:
                frequency[element] = 1
        return frequency