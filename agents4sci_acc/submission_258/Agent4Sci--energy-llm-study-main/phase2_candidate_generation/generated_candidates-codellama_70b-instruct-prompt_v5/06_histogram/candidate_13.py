def solve(input_list):
    unique_elements = list(set(input_list))
    result_dict = {}
    
    for element in unique_elements:
        result_dict[element] = input_list.count(element)
    
    return result_dict