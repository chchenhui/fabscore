# Example usage of 'solve' function:
graph = {
  1: [2, 3],
  2: [4, 5],
  3: [],
  4: [],
  5: []
}
starting_node = 1
result = solve(graph, starting_node)
print("Nodes visited in BFS order starting from node {}: {}".format(starting_node, result))