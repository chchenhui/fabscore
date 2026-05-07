def solve(lst):
  if not all(isinstance(tup, tuple) and len(tup) == 2 and isinstance(tup[0], int) and isinstance(tup[1], str) for tup in lst):
    raise TypeError("Invalid input. All items must be tuples of (int, str).")
  
  counts = {}
  for tup in lst:
    key = tup[0]
    if key not in counts:
      counts[key] = 1
    else:
      counts[key] += 1
  
  return sorted(lst, key=lambda x: (x[0], -counts[x[0]]))