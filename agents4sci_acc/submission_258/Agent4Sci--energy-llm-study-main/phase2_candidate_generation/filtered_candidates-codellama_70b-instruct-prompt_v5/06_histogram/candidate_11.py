def solve(lst):
    freq = {}
    if len(lst) == 0:
        return {}
    for item in lst:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    return freq