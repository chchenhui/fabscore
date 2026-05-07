import csv
def solve(csv_data):
    reader = csv.DictReader(csv_data.splitlines())
    rows = list(reader)
    if not rows:
        raise ValueError("Invalid CSV data")
    return rows