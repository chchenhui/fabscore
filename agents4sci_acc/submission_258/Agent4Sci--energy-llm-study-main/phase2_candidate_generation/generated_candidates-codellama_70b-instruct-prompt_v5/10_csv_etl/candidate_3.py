try:
    header = csv_data.split('\n')[0].split(',')
    rows = csv_data.split('\n')[1:]
except IndexError as e:
    raise ValueError("Invalid CSV data")

result = []
for row in rows:
    values = row.split(',')
    if len(values) != len(header):
        raise ValueError("Invalid CSV data")
    row_dict = {}
    for i, value in enumerate(values):
        row_dict[header[i]] = value
    result.append(row_dict)
return result