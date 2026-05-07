def solve(input_list, k):
    # Sort the list in descending order
    sorted_list = sorted(input_list, reverse=True)

    if not input_list or k <= 0:
        return []
    elif k > len(sorted_list):
        return sorted_list
    else:
        # Return the first k largest numbers in descending order
        return sorted_list[:k]