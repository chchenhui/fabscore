def solve(array):
    try:
        for tup in array:
            if type(tup) != tuple or type(tup[0]) != int or type(tup[1]) != str:
                raise TypeError("Input must be a list of tuples (int, str)")
        return sorted(array, key=lambda x: x[0])  # sort by key in ascending order; stable sort preserves original order for equal keys.
    except Exception as e:
        print(e)