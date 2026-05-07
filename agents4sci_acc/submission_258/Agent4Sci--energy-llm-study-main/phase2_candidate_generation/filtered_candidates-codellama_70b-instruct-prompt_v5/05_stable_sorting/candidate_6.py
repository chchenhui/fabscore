def solve(lst):
    try:
        # Sort the list by key in ascending order while preserving original order for equal keys
        sorted_lst = sorted(lst, key=lambda x: x[0])
        
        # Check if all input elements are tuples of (int, str)
        for element in sorted_lst:
            assert isinstance(element, tuple), f"Input must contain tuples only. Invalid: {element}"
            
            key, value = element
            assert isinstance(key, int), f"Key must be an integer. Invalid key: {key}"
            assert isinstance(value, str), f"Value must be a string. Invalid value: {value}"
        
        return sorted_lst
    except AssertionError as e:
        raise TypeError(str(e)) from None