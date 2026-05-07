def solve(input_list):
    return {i : input_list.count(i) for i in set(input_list)} if input_list else {}