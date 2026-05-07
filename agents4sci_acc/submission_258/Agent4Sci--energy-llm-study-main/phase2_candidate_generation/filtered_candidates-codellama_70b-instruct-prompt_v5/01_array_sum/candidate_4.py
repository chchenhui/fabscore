def solve(list):
    if not list:
        return 0
    sum = 0
    for element in list:
        if type(element) == int or type(element) == float:
            sum += element
        else:
            raise TypeError("Invalid input")
    return sum