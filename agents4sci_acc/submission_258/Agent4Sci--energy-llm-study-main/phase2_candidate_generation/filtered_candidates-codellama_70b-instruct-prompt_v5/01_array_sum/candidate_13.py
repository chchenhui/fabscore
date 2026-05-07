def solve(list):
    for num in list:
        if not (isinstance(num, int) or isinstance(num, float)):
            raise TypeError("Invalid input")
    return sum(list) or 0