def solve(list_of_elements):
    unique = set(list_of_elements) # create set to remove duplicates
    freqs = {} # define empty dictionary
    for element in unique:
        count = list_of_elements.count(element) # count frequency of each element
        freqs[element] = count  # assign to corresponding key in dict
    return freqs