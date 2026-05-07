def solve(input_list):
    try:
        sorted_list = sorted(input_list, key=lambda x: x[0])
    except Exception as e:
        raise TypeError("Invalid input. Expected tuples of (int, str)")
    return sorted_list