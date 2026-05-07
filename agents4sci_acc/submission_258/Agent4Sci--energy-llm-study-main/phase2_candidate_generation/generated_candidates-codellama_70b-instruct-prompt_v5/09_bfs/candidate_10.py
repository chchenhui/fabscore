def solve(graph, start):
    if graph == {}: return [] # Check for empty input graph
    
    visited = set()            # Set to store nodes already visited
    queue = [start]            # Initialize a queue starting with the start node
    order = []                 # List to keep track of the BFS order
        
    while queue:               # Run until the queue is empty (all nodes explored)
        current = queue.pop(0) # Dequeue the first element in the queue
    
        if current not in visited:   # If node is unvisited
            visited.add(current)     # Mark it as visited
            order.append(current)    # Append to BFS order
            
            for neighbor in graph[current]:  # Iterate through neighbors of the current node
                if neighbor not in visited:  # If a neighbor is unvisited
                    queue.append(neighbor)   # Enqueue it to explore later (BFS)
    
    return order               # Return nodes visited in BFS order