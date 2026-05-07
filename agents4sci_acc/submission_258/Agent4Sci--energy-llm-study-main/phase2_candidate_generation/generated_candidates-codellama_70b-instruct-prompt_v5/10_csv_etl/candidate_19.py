import csv
def solve(csv_data):
    reader = csv.DictReader(csv_data.splitlines())
    rows = []
    for row in reader:
        rows.append({k: v for k, v in row.items()})
    return rows