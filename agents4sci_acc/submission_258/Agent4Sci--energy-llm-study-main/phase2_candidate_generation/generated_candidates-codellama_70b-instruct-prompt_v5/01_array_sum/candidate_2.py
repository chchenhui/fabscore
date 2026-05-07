def solve(lst):
    for x in lst:
        if not isinstance(x, (int, float)):
            raise TypeError("All elements must be int or float")
    return sum(lst)