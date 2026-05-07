import csv
def solve(csv_string):
    rows = []
    try:
        reader = csv.DictReader(csv_string.splitlines())
        for row in reader:
            rows.append({k: v for k, v in row.items()})
    except Exception as e:
        raise ValueError("Invalid CSV data")
    return rows