def solve(graph, starting_node):
    visited = set()
    result = []
    
    if not graph or starting_node not in graph:
        return result
    
    queue = [starting_node]
    while queue:
        current_node = queue.pop(0)
        
        if current_node not in visited:
            visited.add(current_node)
            result.append(current_node)
            
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    queue.append(neighbor)
    
    return result