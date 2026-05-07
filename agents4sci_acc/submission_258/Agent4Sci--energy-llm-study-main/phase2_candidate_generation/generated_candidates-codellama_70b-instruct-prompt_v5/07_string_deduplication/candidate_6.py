def solve(strings):
    unique_list = []
    for string in strings:
        if not isinstance(string, str):
            raise TypeError("All elements must be strings")
        if string not in unique_list:
            unique_list.append(string)
    return unique_list