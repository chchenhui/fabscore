def solve(listOfStrings):
    if len(listOfStrings) == 0:
        return []
    else:
        uniqueList = [str]
        for x in listOfStrings:
            if not isinstance(x, str):
                raise TypeError("Invalid input. All elements must be strings.")
            elif x not in uniqueList:
                uniqueList.append(x)
        return uniqueList[1:]