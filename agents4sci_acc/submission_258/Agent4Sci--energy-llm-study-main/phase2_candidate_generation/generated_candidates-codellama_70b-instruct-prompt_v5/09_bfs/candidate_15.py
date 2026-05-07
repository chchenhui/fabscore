def solve(graph, start):
    # Check if the graph is empty or the starting node is not in the graph
    if len(graph) == 0 or start not in graph:
        return []
    # Initialize variables for BFS
    queue = [start]
    visited = set([start])
    result = []
    while queue:
        current_node = queue.pop(0)
        result.append(current_node)
        # Enqueue neighbors of the current node that have not been visited yet
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
    return result