from collections import deque
def solve(graph, start):
    if not graph or not start in graph: return [] # Empty graph or invalid starting node.
    visited = set()  # Keep track of nodes visited
    bfs = deque([start])  # Queue to store the next nodes to visit
    result = []   # List to store the final BFS order.
    
    while bfs:
        current_node = bfs.popleft() # Remove and return an element from left side of the queue.
        
        if not current_node in visited:
            visited.add(current_node)  # Mark node as visited
            result.append(current_node)  # Add node to BFS order list
            
            for neighbor in graph[current_node]:  # Iterate over neighbors of the current node
                if not neighbor in visited:  # If neighbor has not been visited, add it to the queue
                    bfs.append(neighbor)
    
    return result   # Return list containing nodes visited in BFS order