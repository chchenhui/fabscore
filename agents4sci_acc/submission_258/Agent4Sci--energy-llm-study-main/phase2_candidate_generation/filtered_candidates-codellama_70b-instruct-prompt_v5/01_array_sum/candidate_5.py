def solve(numbers):
    # if the list is empty, return zero
    if not len(numbers) > 0:
        return 0

    result = 0
    
    for num in numbers:
      # check if the current element is a number using `isinstance` function. If not, raise TypeError.
      if not isinstance(num, (int, float)):
          raise TypeError("List contains non-numeric elements")
      
      result += num

    return result