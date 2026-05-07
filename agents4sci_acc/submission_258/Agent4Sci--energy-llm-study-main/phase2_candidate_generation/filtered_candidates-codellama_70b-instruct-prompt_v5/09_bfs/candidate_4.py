from collections import deque
def solve(graph, start):
    if not graph or start not in graph:
        return []
    queue = deque()
    visited = set()
    result = []

    queue.append(start)
    visited.add(start)

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result