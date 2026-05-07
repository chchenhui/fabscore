import csv
def solve(csv_string):
    # Check if the CSV string contains at least one line (header row)
    if '\n' not in csv_string:
        raise ValueError("Invalid CSV: Expected at least 1 line")
    
    # Use StringIO to simulate a file-like object
    from io import StringIO
    csvfile = StringIO(csv_string)
    
    rows = []
    reader = csv.DictReader(csvfile)
    
    for row in reader:
        rows.append({k: v for k, v in row.items()})
    
    return rows