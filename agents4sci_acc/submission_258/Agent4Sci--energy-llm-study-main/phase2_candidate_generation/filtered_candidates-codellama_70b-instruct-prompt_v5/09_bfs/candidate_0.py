from collections import deque

def solve(graph: dict, start: int) -> list:
    # Check if the graph is empty or the start node is not valid
    if len(graph) == 0 or start not in graph:
        return []

    result = [start]
    visited = set()
    
    queue = deque([start])
    while queue:
        node = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                result.append(neighbor)
                queue.append(neighbor)
                visited.add(neighbor)

    return result