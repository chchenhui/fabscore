from collections import deque
def solve(graph, start):
    if not graph or start not in graph:
        return [] # Return empty list when graph is empty or starting node not in graph.
    result = [start]
    queue = deque([start]) 
    visited = set()
    
    while queue:
        current_node = queue.popleft()   # Dequeue the current node from front of the queue
        
        if current_node not in visited:
            visited.add(current_node)    # Mark the current node as visited
            
            for neighbor in graph[current_node]:  # Iterate through all neighbors of the current node
                if neighbor not in visited:       # Check if neighbor is already visited
                    queue.append(neighbor)        # Enqueue the neighbor to be processed later
        
        result += [neighbor]              # Append the neighbor to the BFS traversal path    
    return result                         # Return the list of nodes visited in BFS order starting from the given node