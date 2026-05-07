def solve(strings):
    if strings == []:
        return []
    else:
        for string in strings:
            if type(string) != str:
                raise TypeError("All elements must be of type 'str'")
        unique_list = []
        [unique_list.append(x) for x in strings if x not in unique_list]
    return unique_list