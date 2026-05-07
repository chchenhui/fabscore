def solve(input_list):
    seen = set()
    output = []
    for item in input_list:
        if not isinstance(item, str):
            raise TypeError("All elements must be strings.")
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output