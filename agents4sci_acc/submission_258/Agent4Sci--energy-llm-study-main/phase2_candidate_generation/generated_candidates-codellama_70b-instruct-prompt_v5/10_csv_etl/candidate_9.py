import csv
def solve(csv):
    try:
        rows = list(csv.reader(csv))
        header, data = [row.strip() for row in rows[0]], [[col.strip() for col in row] for row in rows[1:]]
        return [{key: value for key, value in zip(header, row)} for row in data]
    except Exception as e:
        raise ValueError("Invalid CSV provided") from e