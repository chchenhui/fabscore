def solve(graph, start):
    # Check if graph is empty or node not in graph.
    if len(graph) == 0 or start not in graph:
        return []   # Return an empty list.
    
    queue = [start]  # Initialize queue with the starting node.
    visited = set()  # Keep track of visited nodes.
    result = []      # List to store the BFS order.
    
    while queue:
        current_node = queue.pop(0)  # Dequeue a node from the front of the queue.
        
        if current_node not in visited:
            visited.add(current_node)   # Mark the node as visited.
            result.append(current_node) # Add it to the result list.
            
            for neighbor in graph[current_node]:  # Iterate through all neighbors of current node.
                if neighbor not in visited:
                    queue.append(neighbor)   # Enqueue unvisited neighbors.
    
    return result