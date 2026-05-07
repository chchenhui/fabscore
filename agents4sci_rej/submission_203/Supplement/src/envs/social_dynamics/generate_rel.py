import csv

def generate_grid_connections(rows=10, cols=10):
    """
    Generate connections between adjacent points in a grid.
    Each point is connected to its neighbors above, below, left, and right.
    """
    connections = []
    
    # For each cell in the grid
    for row in range(rows):
        for col in range(cols):
            # Calculate the current cell ID
            current_id = row * cols + col
            
            # Connect to the cell above (if it exists)
            if row > 0:
                top_id = (row - 1) * cols + col
                connections.append((current_id, top_id, "unidirectional"))
                connections.append((top_id, current_id, "unidirectional"))
            
            # Connect to the cell below (if it exists)
            if row < rows - 1:
                bottom_id = (row + 1) * cols + col
                connections.append((current_id, bottom_id, "unidirectional"))
                connections.append((bottom_id, current_id, "unidirectional"))
            
            # Connect to the cell to the left (if it exists)
            if col > 0:
                left_id = row * cols + (col - 1)
                connections.append((current_id, left_id, "unidirectional"))
                connections.append((left_id, current_id, "unidirectional"))
            
            # Connect to the cell to the right (if it exists)
            if col < cols - 1:
                right_id = row * cols + (col + 1)
                connections.append((current_id, right_id, "unidirectional"))
                connections.append((right_id, current_id, "unidirectional"))
    
    return connections

def save_to_csv(connections, filename="profile/data/Relationship.csv"):
    """Save the connections to a CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["source_id", "target_id", "direction"])
        for connection in connections:
            writer.writerow(connection)
    
    print(f"Saved {len(connections)} connections to {filename}")

def main():
    # Generate connections for a 10x10 grid
    connections = generate_grid_connections(10, 10)
    
    # Remove duplicates (in case we counted some connections twice)
    unique_connections = list(set(connections))
    
    # Save to CSV
    save_to_csv(unique_connections)

if __name__ == "__main__":
    main()