import csv
def solve(csvData):
  try:
    # Parse the CSV string into a list of dictionaries
    result = []
    reader = csv.DictReader(csvData)
    for row in reader:
      result.append(row)
    return result
  except Exception as e:
    raise ValueError("Invalid CSV data.") from e