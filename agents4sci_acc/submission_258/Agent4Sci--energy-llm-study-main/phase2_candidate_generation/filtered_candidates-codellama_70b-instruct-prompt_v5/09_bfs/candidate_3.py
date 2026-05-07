from collections import deque

def solve(graph, start):
    if not graph or start not in graph:
        return []  # Return empty list if the graph is empty or the starting node is not in the graph
    
    visited = set()  # Set to keep track of visited nodes
    queue = deque([start])  # Initialize a queue with the starting node
    result = [start]

    while queue:
        node = queue.popleft()  # Dequeue the next node
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)  # Mark the neighbor as visited
                queue.append(neighbor)  # Enqueue the neighbor for processing
                result.append(neighbor)
    return result