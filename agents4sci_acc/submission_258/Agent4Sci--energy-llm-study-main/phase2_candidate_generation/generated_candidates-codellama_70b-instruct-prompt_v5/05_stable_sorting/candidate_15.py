def solve(list1):
    # check the data types of each tuple in list1
    if not all([isinstance(x[0], int) and isinstance(x[1], str) for x in list1]):
        raise TypeError("Input must be a list of (int, str) tuples")
    
    # sort the list using sorted() function 
    return sorted(list1, key=lambda tup: tup[0])