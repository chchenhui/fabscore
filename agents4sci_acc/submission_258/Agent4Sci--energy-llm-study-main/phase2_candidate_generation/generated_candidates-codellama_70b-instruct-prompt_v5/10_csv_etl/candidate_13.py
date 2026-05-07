import csv
def solve(csv_string):
    csv_reader = csv.DictReader(csv_string.splitlines())
    result = list()
    for row in csv_reader:
        result.append(row)
    return result