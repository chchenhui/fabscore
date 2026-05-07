def solve(input_list):
    if len(input_list) == 0:
        return []
    for string in input_list:
        if not isinstance(string, str):
            raise TypeError("Invalid input. All elements must be strings.")

    unique = set()
    output_list = []
    for string in input_list:
        if string not in unique:
            unique.add(string)
            output_list.append(string)

    return output_list