def solve(strings):
    unique_dict = {}
    for string in strings:
        if not isinstance(string, str):
            raise TypeError("Input must be a list of strings.")
        unique_dict[string] = None
    return list(unique_dict.keys())