def solve(elements):
    count = {}
    for e in elements:
        if e not in count:
            count[e] = 1
        else:
            count[e] += 1
    return count