def solve(lst):
    try:
        if lst: # Checking if list is not empty
            return sum(lst)
        else:   # Handling when the list is empty
            return 0
    except TypeError as e:
        raise TypeError("At least one element in the list is not a number")