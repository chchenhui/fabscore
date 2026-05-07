def solve(lst):
    # return empty dictionary if the list is empty
    if len(lst) == 0:
        return {}
    # create a frequency map using dictionary comprehension
    freq = {element: lst.count(element) for element in set(lst)}
    return freq