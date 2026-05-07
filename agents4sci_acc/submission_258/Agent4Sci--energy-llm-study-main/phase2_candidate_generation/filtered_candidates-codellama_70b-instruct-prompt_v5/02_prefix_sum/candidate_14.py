from itertools import accumulate
def solve(integers):
    # Checking if the input argument is a list or not
    if not type(integers) == type([]):
        raise TypeError("The given argument is not a list")
    
    # Iterating over each element of the list to check if it's an integer or not, raising TypeError for non-integer elements.
    for item in integers:
        if not type(item) == type(int()):
            raise TypeError("One of the elements is not an integer")
            
    # Calculating and returning prefix sums using accumulate function from itertools module (can be replaced with own implementation if needed)
    return list(accumulate(integers))