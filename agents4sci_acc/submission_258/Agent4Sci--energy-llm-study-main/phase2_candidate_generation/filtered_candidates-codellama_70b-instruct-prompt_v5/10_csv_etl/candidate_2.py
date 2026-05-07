import csv

def solve(csv_string):
    """Parse CSV string and return a list of dictionaries."""
    try:
        reader = csv.DictReader(csv_string.splitlines())
        result = [row for row in reader]
        return result
    except Exception as e:
        raise ValueError("Invalid CSV") from e