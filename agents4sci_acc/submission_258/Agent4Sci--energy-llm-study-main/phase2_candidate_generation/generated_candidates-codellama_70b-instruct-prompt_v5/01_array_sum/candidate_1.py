def solve(numbers):
    try:
        if len(numbers) == 0:
            return 0
        else:
            return sum(numbers)
    except TypeError as e:
        raise TypeError("List contains invalid elements") from e