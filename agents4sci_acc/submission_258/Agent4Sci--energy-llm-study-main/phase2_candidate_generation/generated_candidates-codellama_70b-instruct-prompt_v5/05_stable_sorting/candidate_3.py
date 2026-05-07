from typing import List, Tuple
def solve(input_list: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    # Check if all elements are tuples of (int, str)
    if any([type(element) != tuple for element in input_list]) or any([type(element[0]) != int or type(element[1]) != str for element in input_list]):
        raise TypeError("Invalid input: not all elements are tuples of (int, str)")
    # Sort the list by key (the first element) using a stable sorting algorithm
    output_list = sorted(input_list, key=operator.itemgetter(0))
    return output_list
  
# Test cases:
test1 = [(4, "hello"), (2, "world"), (3, "python")]
result1 = solve(test1)
print("Input list:", test1)
print("Output list:", result1)

test2 = [(-5, "negative"), (0, "zero"), (4, "positive")]
result2 = solve(test2)
print("\nInput list:", test2)
print("Output list:", result2)