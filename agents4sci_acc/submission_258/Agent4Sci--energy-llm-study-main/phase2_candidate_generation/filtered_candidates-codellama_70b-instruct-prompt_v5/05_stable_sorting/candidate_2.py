def solve(input_list):
    if not input_list: return [] # Empty list, so nothing to sort.

    for key, value in input_list:
        if not isinstance(key, int) or not isinstance(value, str):
            raise TypeError("Invalid input format. Expected [(int, str), ...]")

    return sorted(input_list, key=lambda x: x[0]) # Sort by integer keys in ascending order, stable sort