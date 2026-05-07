from collections import deque
def solve(graph, starting_node):
    if len(graph) == 0 or starting_node not in graph:
        return []
    
    visited = set()
    queue = deque([starting_node])
    result = [starting_node]
    
    while queue:
        node = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                result.append(neighbor)
    
    return result