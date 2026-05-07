import csv

def solve(csv_data):
    """Parse a string containing CSV data and return a list of dictionaries.

    Parameters:
        csv_data (str): A string representing CSV data with a header row.

    Returns:
        dict_rows (list): A list of dictionaries, one for each row in the CSV data.
            Each dictionary has keys corresponding to the column names from the header.

    Raises:
        ValueError: If the CSV is invalid.
    """

    # Create a StringIO object and read it as csvfile using csv reader
    try:
        with io.StringIO(csv_data) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            dict_rows = [row for row in reader]
    except Exception:
        raise ValueError("Invalid CSV data.") from None

    return dict_rows