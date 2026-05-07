import csv

def solve(csv_string):
    """Parses the given string as CSV data with headers.

    Args:
        csv_string (str): A string containing CSV-formatted data, including a header row.

    Returns:
        list[dict]: The parsed CSV rows, each represented by a dictionary whose keys are column names and values are entries for that row.

    Raises:
       ValueError: If the input CSV data is invalid.
    """
    
    # Parse the CSV string with the built-in csv module
    reader = csv.DictReader(csv_string.splitlines())
    try:
        rows = list(reader)
    except Exception as e:
        raise ValueError("Invalid CSV data.") from e

    return rows