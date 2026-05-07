import csv
def solve(csv_string):
    # Initialize an empty result list
    results = []
    
    # Use 'StringIO' to simulate file input from the given string
    f = StringIO(csv_string)
    
    # Create a CSV reader object with 'f' as the source
    csv_reader = csv.reader(f, delimiter=',')
    
    # Read the header row and store it in a variable
    headers = next(csv_reader)
    
    for row in csv_reader:
        if len(row) != len(headers):
            raise ValueError("Invalid CSV data - number of columns does not match the header")
        
        # Convert each row into a dictionary with headers as keys and values as entries
        result_dict = {}
        for i, value in enumerate(row):
            result_dict[headers[i]] = value
        
        results.append(result_dict)
    
    return results