def solve(input_list):
    if len(input_list) == 0:
        return []

    for element in input_list:
        if type(element) != str:
            raise TypeError("All elements must be strings.")

    unique_values = set()
    result = []
    for value in input_list:
        if value not in unique_values:
            unique_values.add(value)
            result.append(value)

    return result