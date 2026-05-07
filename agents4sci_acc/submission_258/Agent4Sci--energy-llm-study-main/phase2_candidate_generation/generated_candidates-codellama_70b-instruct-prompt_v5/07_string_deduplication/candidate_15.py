def solve(strings):
    # If list is empty return an empty list
    if len(strings) == 0:
        return []

    # Initialize a set to keep track of unique strings
    seen = set()
    # Initialize the result list for non-duplicate strings
    output = []

    # Iterate through each string in the input list
    for s in strings:
      # Check if the string is already present in the 'seen' set
        if not isinstance(s, str):
            raise TypeError("The input list must contain only strings.")

        # If it is already seen, skip to the next iteration
        if s in seen:
            continue

        # Otherwise, add the string to both the 'seen' set and the result list
        seen.add(s)
        output.append(s)

    return output