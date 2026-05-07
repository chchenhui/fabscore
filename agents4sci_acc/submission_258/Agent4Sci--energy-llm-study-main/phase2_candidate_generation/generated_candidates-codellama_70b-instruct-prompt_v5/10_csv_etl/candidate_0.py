import csv

def solve(data):
    try:
        reader = csv.DictReader(StringIO(data))
        return [row for row in reader]
    except Exception as e:
        raise ValueError("Invalid CSV data.") from e