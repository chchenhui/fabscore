def solve(graph, start_node):
    if len(graph) == 0 or start_node not in graph:
        return []

    visited = set()
    queue = [start_node]
    result = []
    
    while queue:
        current_node = queue.pop(0)
        
        if current_node not in visited:
            visited.add(current_node)
            result.append(current_node)
            
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    queue.append(neighbor)
    
    return result