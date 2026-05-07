def solve(inputList):
    # check if the list is empty
    if len(inputList) == 0:
        return {}
    else:
        frequencyMap = {}
        for item in inputList:
            if item not in frequencyMap:
                frequencyMap[item] = 1
            else:
                frequencyMap[item] += 1
        return frequencyMap