def solve(csv):
    import io, csv
    
    try:
        reader = csv.DictReader(io.StringIO(csv))
        return list(reader)
    except Exception as e:
        raise ValueError("Invalid CSV string.") from e