def solve(graph: dict, start: str):
    if not graph or start not in graph:
        return []  # Handle empty graph or non-existent start node

    queue = [start]  # Initialize BFS queue with the starting node
    visited = set()  # Keep track of visited nodes
    bfs_order = []   # List to store the order in which nodes were visited during BFS traversal

    while queue:
        current_node = queue.pop(0)  # Dequeue a node from the front of the queue
        if current_node not in visited:
            visited.add(current_node)   # Mark the current node as visited
            bfs_order.append(current_node)  # Add it to the list of visited nodes
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    queue.append(neighbor)  # Enqueue unvisited neighbors

    return bfs_order