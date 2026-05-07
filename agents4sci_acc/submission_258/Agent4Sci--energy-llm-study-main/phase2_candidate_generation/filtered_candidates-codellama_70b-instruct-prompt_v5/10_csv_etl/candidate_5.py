import csv

def solve(csv_data):
    """Parse the given CSV data and return a list of dictionaries."""
    try:
        # Use the 'cvs' module to parse the string into rows and columns
        reader = csv.reader(csv_data.splitlines())

        # Obtain the column names from the header row
        keys = next(reader)
        
        # Create a list of dictionaries (one per row), with each dictionary having keys as column names and values as row entries
        result = []
        for row in reader:
            if len(row) != len(keys):
                raise ValueError("Invalid CSV data: number of columns doesn't match the header")
            
            # Convert the row into a dictionary using zip() to associate keys and values
            result.append({k: v for k, v in zip(keys, row)})
        
        return result
    
    except csv.Error as e:
        raise ValueError("Invalid CSV data") from e