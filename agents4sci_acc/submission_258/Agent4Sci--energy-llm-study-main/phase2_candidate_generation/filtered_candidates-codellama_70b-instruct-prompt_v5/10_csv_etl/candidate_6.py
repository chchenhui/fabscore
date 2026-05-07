import csv

def solve(csv_string):
    # Create a list to store each row as a dictionary
    rows = []
    
    # Use StringIO to handle data from memory
    import io
    csvfile = io.StringIO(csv_string)
    
    # Read the CSV data using the 'DictReader' class, which will return OrderedDicts
    reader = csv.DictReader(csvfile)
    
    try:
        for row in reader:
            rows.append(dict(row))  # Convert each row to a regular dictionary and add it to the list
        return rows
    except Exception as e:  # Catch any error while parsing the CSV data
        raise ValueError("Invalid CSV string provided.") from e