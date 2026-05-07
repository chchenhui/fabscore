def solve(numbers):
    try:
        # Check if the argument passed to the function is a list.
        assert isinstance(numbers, list), "Argument not a list."
        # Return 0 if the list is empty.
        if len(numbers) == 0:
            return 0
        return sum(numbers)
    except Exception as e:
        print("Error:", str(e))