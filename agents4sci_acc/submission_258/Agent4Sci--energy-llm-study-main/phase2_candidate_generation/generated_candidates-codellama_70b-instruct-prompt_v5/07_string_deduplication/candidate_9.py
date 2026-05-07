def solve(input_list):
    if not input_list:
        return []  # Return an empty list if the input is empty.

    for element in input_list:
        if not isinstance(element, str):
            raise TypeError("All elements must be strings.")

    output = []
    seen = set()  # Use a set to keep track of unique values.

    for value in input_list:
        if value not in seen:
            output.append(value)  # Add the current value to the result list.
            seen.add(value)       # Mark it as seen to avoid duplicates.

    return output