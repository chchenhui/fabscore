def solve(list_of_numbers):
    try:
        return sum(list_of_numbers) if len(list_of_numbers)>0 else 0
    except TypeError as e:
        raise TypeError("List contains non-numeric elements")