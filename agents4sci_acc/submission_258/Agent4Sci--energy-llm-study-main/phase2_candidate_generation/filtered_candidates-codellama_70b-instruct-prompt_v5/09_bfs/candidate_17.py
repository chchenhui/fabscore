class Graph:
    def __init__(self):
        self.graph = {}  # Dictionary to represent graph

    def addEdge(self, node1, node2):
        if node1 in self.graph:
            self.graph[node1].append(node2)
        else:
            self.graph[node1] = [node2]

def solve(adjacency_list, start_node):
    visited = set()  # Set to keep track of visited nodes.
    queue = []  # Initialize a queue
    
    if len(adjacency_list) == 0 or start_node not in adjacency_list:
        return []

    bfs_order = [start_node]
    queue.append(start_node)
    visited.add(start_node)
    
    while queue:
        current_node = queue[0]  # Dequeue the first element
        
        for neighbor in adjacency_list[current_node]:
            if neighbor not in visited:
                bfs_order.append(neighbor)
                visited.add(neighbor)
                queue.append(neighbor)  # Enqueue the neighbor
        
        queue.pop(0)  # Remove the first element from the queue
    
    return bfs_order