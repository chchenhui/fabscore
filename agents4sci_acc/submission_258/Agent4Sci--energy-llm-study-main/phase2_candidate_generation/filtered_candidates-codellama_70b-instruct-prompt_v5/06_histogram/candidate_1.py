def solve(list):
    if len(list) == 0:
        return {} # return an empty dictionary for an empty list
    else:
        frequency = {}
        for item in list:
            if item in frequency:
                frequency[item] += 1
            else:
                frequency[item] = 1
        return frequency