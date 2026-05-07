def solve(graph, start):
    # Check if the graph is empty or the starting node is not in the graph
    if not graph or start not in graph:
        return []

    visited = set()  # Set to keep track of visited nodes.
    queue = [start]  # Initialize a queue with the starting node.
    bfs_order = []   # List to store visited nodes in BFS order.

    while queue:
        current_node = queue.pop(0)
        if current_node not in visited:
            visited.add(current_node)  # Mark the current node as visited.
            bfs_order.append(current_node)  # Add the current node to BFS order list.
            for neighbor in graph[current_node]:   # Iterate through neighbors of the current node.
                if neighbor not in visited:
                    queue.append(neighbor)  # Append unvisited neighbors to the queue.

    return bfs_order