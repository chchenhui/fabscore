def solve(strings):
    # Check if all elements in the input list are strings
    for item in strings:
        if type(item) != str:
            raise TypeError("All elements must be of type string")

    # Use a set to keep track of unique strings seen so far
    unique_strings = set()

    # Initialize an empty list to store the output
    ordered_unique_strings = []

    # Iterate through each string in the input list and add it to the output if it hasn't been seen before.
    for s in strings:
        if s not in unique_strings:
            unique_strings.add(s)
            ordered_unique_strings.append(s)

    return ordered_unique_strings