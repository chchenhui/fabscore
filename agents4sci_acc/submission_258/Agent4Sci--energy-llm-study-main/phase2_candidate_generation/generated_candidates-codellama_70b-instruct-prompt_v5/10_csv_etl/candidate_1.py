import csv
def solve(csv):
    """Parse the CSV string and return a list of dictionaries."""
    with StringIO(csv) as f:
        reader = csv.DictReader(f, delimiter=',')
        try:
            return [dict(row) for row in reader]
        except ValueError as e:
            raise ValueError("Invalid CSV") from e