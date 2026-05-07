def solve(input_list):
    # Create an empty dictionary to store the frequency of each unique element
    freq = {}

    # Iterate through all elements in the input list
    for elem in input_list:
        if elem in freq:
            # If the current element is already present in the dictionary, increment its frequency by 1
            freq[elem] += 1
        else:
            # Otherwise, initialize the frequency to 1
            freq[elem] = 1

    return freq