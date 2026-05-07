import queue
def solve(graph, start):
    if len(graph) == 0 or start not in graph:
        return []

    result = []
    seen = set([start])
    q = queue.Queue()
    q.put(start)

    while not q.empty():
        node = q.get()
        result.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                q.put(neighbor)
            
    return result