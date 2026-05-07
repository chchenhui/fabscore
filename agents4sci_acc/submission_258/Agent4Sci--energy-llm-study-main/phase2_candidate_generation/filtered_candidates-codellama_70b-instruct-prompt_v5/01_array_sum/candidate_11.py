def solve(numbers):
    # Initialize an empty result variable to store the sum
    result = 0
    for num in numbers:
        try:
            result += float(num)
        except ValueError as e:
            raise TypeError("Element {0} is not a number.".format(str(num))) from None
    return result