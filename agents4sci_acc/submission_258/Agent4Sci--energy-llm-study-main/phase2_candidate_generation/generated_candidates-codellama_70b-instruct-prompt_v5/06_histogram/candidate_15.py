def solve(list_of_elements):
    frequency = {}
    for element in list_of_elements:
        if element in frequency:
            frequency[element] += 1
        else:
            frequency[element] = 1
    return frequency