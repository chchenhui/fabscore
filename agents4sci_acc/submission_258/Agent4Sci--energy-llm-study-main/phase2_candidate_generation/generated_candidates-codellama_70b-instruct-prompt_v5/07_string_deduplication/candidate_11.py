def solve(strings):
    if not isinstance(strings, list): # check if input is a list
        raise TypeError("Input must be a list") 
    unique = [] # initialize an empty list for unique strings
    seen = set() # create a set to store the strings that have been added
    for s in strings: 
        if not isinstance(s, str): # check if each element is a string
            raise TypeError("All elements must be strings")  
        if s not in seen: # check if the current string has been previously added
            unique.append(s) # add it to the list of unique strings
            seen.add(s) 
    return unique  # return the list with duplicates removed