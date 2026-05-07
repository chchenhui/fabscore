def solve(list1):
    if list1 == []:
        return []
    output = []
    for x in list1:
        if not isinstance(x, str):
            raise TypeError('Invalid input')
        elif x not in output:
            output.append(x)
    return output