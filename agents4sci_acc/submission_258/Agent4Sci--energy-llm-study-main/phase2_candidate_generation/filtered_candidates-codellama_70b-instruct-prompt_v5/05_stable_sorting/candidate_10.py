# Function to sort a list of tuples with (key, value) pairs by key in ascending order
def solve(list_tuples):
    # Check if all elements are tuples of (int, str)
    for element in list_tuples:
        if not isinstance(element[0], int) or not isinstance(element[1], str):
            raise TypeError("Input elements must be tuples of (int, str)")
    
    # Sort the list by key in ascending order, preserving original order for equal keys.
    sorted_list = sorted(list_tuples, key=lambda x:x[0])
    return sorted_list