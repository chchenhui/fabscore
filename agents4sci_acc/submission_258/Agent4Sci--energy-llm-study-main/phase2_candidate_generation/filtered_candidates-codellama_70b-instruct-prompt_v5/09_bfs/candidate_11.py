def solve(graph, start):
    visited = []
    queue = []
    
    if graph and (start in graph.keys()):
        queue.append(start)
        
        while queue:
            node = queue.pop(0)
            
            if not node in visited:
                visited.append(node)
                
            for neighbor in graph[node]:
                if not neighbor in visited:
                    queue.append(neighbor)
        
        return visited
    
    else:
        return []