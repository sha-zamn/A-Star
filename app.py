# --- IMPORTS ---
import gradio as gr          # Library for creating the web interface (UI)
import heapq                 # Library for Priority Queue (ESSENTIAL for Best-First Search)
import matplotlib            # Library for plotting/graphing
matplotlib.use('Agg')        # CRITICAL: Tells matplotlib to run in 'server mode' (no display screen needed)
import matplotlib.pyplot as plt # Specific plotting tools
import numpy as np           # Library for efficient numerical operations (arrays)

# --- A* SEARCH ALGORITHM ---
def a_star_search(grid, start, goal, heuristic_type):
    # Get dimensions of the map (rows, columns)
    rows, cols = len(grid), len(grid[0])
    
    # OPEN LIST: Priority Queue to store nodes to be explored
    # Format: (priority_value, current_cost, current_position)
    # heapq ensures the node with the LOWEST priority_value is popped first (Best-First)
    # Initial value: f=0, g=0, position=start
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, goal, heuristic_type), 0, start))
    
    # Dictionary to track where we came from (for rebuilding the path later)
    came_from = {}
    
    # Dictionary to track the cost to reach each node (g-score)
    # g_score[start] = 0 because cost to reach start is zero
    g_score = {start: 0}
    
    # CLOSED SET: Tracks nodes we have already fully explored
    # Prevents the algorithm from going in circles or re-processing nodes
    closed_set = set()
    
    # Counter to measure computational effort (nodes expanded)
    # This is key for demonstrating efficiency in your presentation
    nodes_expanded = 0

    # MAIN SEARCH LOOP: Runs until there are no more nodes to explore
    while open_list:
        # Get the node with the lowest f-score (g + h) from the priority queue
        # This is the "Best-First" decision: picking the most promising node
        current_f, current_g, current = heapq.heappop(open_list)
        
        # Increment counter (used to demonstrate efficiency in your presentation)
        nodes_expanded += 1
        
        # Skip if we already processed this node (avoids cycles)
        if current in closed_set: 
            continue
        
        # GOAL CHECK: If we reached the target, stop and return result
        if current == goal:
            return reconstruct_path(came_from, current), nodes_expanded
        
        # Mark current node as processed
        closed_set.add(current)
        
        # NEIGHBOR EXPANSION: Check Up, Down, Left, Right (4-directional movement)
        # This matches the Manhattan distance logic
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            # Calculate neighbor coordinates
            neighbor = (current[0] + dx, current[1] + dy)
            
            # BOUNDARY & OBSTACLE CHECK: 
            # 1. Is neighbor inside the map? (0 <= neighbor < rows/cols)
            # 2. Is neighbor NOT an obstacle? (grid value == 0)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0]][neighbor[1]] == 0:
                
                # Calculate tentative cost to reach this neighbor (current cost + 1 step)
                # In grid search, each step usually costs 1
                tentative_g = current_g + 1
                
                # If this path to neighbor is better than any previous path found...
                # (Either we haven't seen it, or we found a cheaper way to get there)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Record this new best path
                    g_score[neighbor] = tentative_g
                    
                    # Calculate Heuristic (h-score) based on user selection (Manhattan vs. Euclidean)
                    h = heuristic(neighbor, goal, heuristic_type)
                    
                    # STANDARD A* EQUATION: f = g + h
                    # This determines the priority in the Best-First Queue
                    f_score = tentative_g + h
                    
                    # Add neighbor to Priority Queue to be explored later
                    heapq.heappush(open_list, (f_score, tentative_g, neighbor))
                    
                    # Record parent for path reconstruction
                    came_from[neighbor] = current
    
    # If loop finishes without finding goal, no path exists
    return None, nodes_expanded

# --- HEURISTIC FUNCTIONS ---
def heuristic(a, b, type):
    # Manhattan Distance: Best for grid movement (only Up/Down/Left/Right)
    # Formula: |x1 - x2| + |y1 - y2|
    # Admissible for 4-directional grids (never overestimates)
    if type == "Manhattan": 
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    # Euclidean Distance: Straight-line distance (as the crow flies)
    # Formula: sqrt((x1 - x2)^2 + (y1 - y2)^2)
    # Also admissible for 4-directional, but underestimates MORE than Manhattan
    else: 
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

# --- PATH RECONSTRUCTION ---
def reconstruct_path(came_from, current):
    # Start from the goal and work backwards to start using parent pointers
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    # Reverse the list so it goes Start -> Goal
    return total_path[::-1]

# --- MAIN FUNCTION (Called by UI) ---
def run_astar(heuristic_type, obstacle_density):
    max_attempts = 5  # Safety net: Try to generate a solvable map up to 5 times
    grid = None
    path = None
    nodes = 0
    
    # AUTO-RETRY LOOP: Prevents "No Path Found" errors due to bad random generation
    for attempt in range(max_attempts):
        # 1. Generate Empty Map (25x25 grid)
        grid = [[0 for _ in range(25)] for _ in range(25)]
        
        # 2. Add Random Obstacles based on slider density
        for i in range(25):
            for j in range(25):
                # np.random.random() gives 0.0 to 1.0. If less than density%, make it a wall (1)
                if np.random.random() < obstacle_density/100:
                    grid[i][j] = 1
        
        # 3. Ensure Start (0,0) and Goal (24,24) are never blocked
        grid[0][0] = 0
        grid[24][24] = 0
        
        start = (0, 0)
        goal = (24, 24)
        
        # 4. Run A* Algorithm
        path, nodes = a_star_search(grid, start, goal, heuristic_type)
        
        # 5. If path found, break the retry loop
        if path:
            break
    
    # --- VISUALIZATION ---
    # Create a 5x5 inch plot
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Display the grid map (0=White, 1=Black)
    ax.imshow(np.array(grid), cmap='binary')
    
    if path:
        # Extract x and y coordinates for plotting
        path_y, path_x = zip(*path)
        
        # Draw the path in Red
        ax.plot(path_x, path_y, 'r-', linewidth=2, label='Path')
        
        # Draw Start (Green) and Goal (Blue)
        ax.plot(start[1], start[0], 'go', markersize=10, label='Start')
        ax.plot(goal[1], goal[0], 'bo', markersize=10, label='Goal')
        
        # Prepare success message with metrics
        result = f"✅ Path Found!\nNodes Expanded: {nodes}\nPath Length: {len(path)}"
    else:
        # Prepare failure message
        result = "❌ No Path Found (Map too crowded)\nTry lowering density."
        ax.set_title("Blocked Map")
    
    # Add legend and grid lines
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    
    # Return the plot image and the text result to the UI
    return fig, result

# --- GRADIO USER INTERFACE ---
# Creates a custom layout block
with gr.Blocks(title="A* Path Planning", css=".gradio-container {max-width: 600px !important;}") as demo:
    gr.Markdown("# 🧠 A* Heuristic Lab")
    gr.Markdown("Compare Manhattan vs. Euclidean Heuristics")
    
    with gr.Row():
        with gr.Column():
            # Input 1: Dropdown to choose Heuristic
            heuristic_dropdown = gr.Dropdown(
                choices=["Manhattan", "Euclidean"],
                value="Manhattan",
                label="Heuristic Function (h(n))"
            )
            # Input 2: Slider to choose Obstacle Density
            density_slider = gr.Slider(
                minimum=10, maximum=30, value=15, step=5,
                label="Obstacle Density (%)"
            )
            # Input 3: Button to trigger the search
            run_button = gr.Button("Run A* Search", variant="primary")
        
        with gr.Column():
            # Output 1: The Matplotlib Plot
            output_plot = gr.Plot(label="Path Visualization")
            # Output 2: The Text Results
            output_text = gr.Textbox(label="Results")
    
    # Connect the Button Click to the Function
    # When button is clicked -> run run_astar -> send inputs -> get outputs
    run_button.click(
        fn=run_astar,
        inputs=[heuristic_dropdown, density_slider],
        outputs=[output_plot, output_text]
    )

# Launch the web app
if __name__ == "__main__":
    demo.launch()