import csv
def solve(csv_string):
    try:
        reader = csv.DictReader(csv_string.splitlines())
        return list(reader)
    except Exception as e:
        raise ValueError("Invalid CSV") from e