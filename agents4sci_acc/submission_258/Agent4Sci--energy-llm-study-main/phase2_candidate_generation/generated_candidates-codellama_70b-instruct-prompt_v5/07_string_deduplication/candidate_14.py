def solve(input_list):
    # Check if the input is not empty and has only strings as elements.
    if len(input_list) == 0 or any(not isinstance(element, str) for element in input_list):
        return []

    # Use a set to keep track of unique elements seen so far.
    seen = set()
    output_list = []

    # Iterate through the input list and add only unique elements to the output list while preserving their order.
    for element in input_list:
        if element not in seen:
            seen.add(element)
            output_list.append(element)

    return output_list