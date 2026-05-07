def solve(input_list):
    # Create an empty dictionary to store the frequency of elements
    frequency = {}
    # Iterate through each element in the input list
    for item in input_list:
        if item in frequency:
            # If the item is already present in the dictionary, increment its frequency by 1
            frequency[item] += 1
        else:
            # If the item is not present in the dictionary, add it to the dictionary with a frequency of 1.
            frequency[item] = 1
    return frequency