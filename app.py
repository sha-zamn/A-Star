# ==============================================================================
# COURSE: AI 211 - A* Search in Best-First Search Framework
# PROJECT: Interactive A* Heuristic Lab (Controlled Experiment)
#
# DESCRIPTION:
# This application demonstrates the A* Search Algorithm, a specific instance of 
# Best-First Search. It allows users to compare two common heuristic functions 
# (Manhattan vs. Euclidean) on deterministic maps to isolate the impact of h(n).
#
# TECHNICAL STACK:
# - Gradio: For web-based interactive UI
# - Heapq: For efficient Priority Queue (O(log n) extraction)
# - Matplotlib: For path visualization
# - Numpy: For grid management and random seed control
# ==============================================================================

import gradio as gr          # Library for creating the web interface (UI)
import heapq                 # Library for Priority Queue (ESSENTIAL for Best-First Search)
import matplotlib            # Library for plotting/graphing
matplotlib.use('Agg')        # CRITICAL: Tells matplotlib to run in 'server mode' (no display screen needed)
import matplotlib.pyplot as plt # Specific plotting tools
import numpy as np           # Library for efficient numerical operations (arrays)

# ==============================================================================
# CORE ALGORITHM: A* Search
# ==============================================================================
def a_star_search(grid, start, goal, heuristic_type):
    """
    Implements the standard A* Search Algorithm.
    Equation: f(n) = g(n) + h(n)
    
    Args:
        grid (list): 2D list representing the map (0=Free, 1=Obstacle)
        start (tuple): (row, col) starting position
        goal (tuple): (row, col) target position
        heuristic_type (str): 'Manhattan' or 'Euclidean'
    
    Returns:
        path (list): List of coordinates from start to goal (if found)
        nodes_expanded (int): Count of nodes processed (measure of efficiency)
    """
    # Get dimensions of the map (rows, columns)
    rows, cols = len(grid), len(grid[0])
    
    # --- OPEN LIST (Priority Queue) ---
    # This is the 'Frontier' in Best-First Search.
    # Format: (priority_value, current_cost, current_position)
    # heapq ensures the node with the LOWEST priority_value is popped first.
    # This implements the 'Best-First' selection strategy.
    # Initial value: f=0+heuristic, g=0, position=start
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, goal, heuristic_type), 0, start))
    
    # --- CAME_FROM (Dictionary) ---
    # Tracks the parent of each node. Used to reconstruct the path once goal is found.
    came_from = {}
    
    # --- G_SCORE (Dictionary) ---
    # Tracks the actual cost from Start to Current Node (g(n)).
    # Initialized to 0 for start node.
    g_score = {start: 0}
    
    # --- CLOSED SET (Set) ---
    # Tracks nodes we have already fully explored.
    # Prevents the algorithm from going in circles or re-processing nodes.
    closed_set = set()
    
    # Counter to measure computational effort (nodes expanded).
    # This is key for demonstrating efficiency in your presentation.
    nodes_expanded = 0

    # --- MAIN SEARCH LOOP ---
    # Runs until there are no more nodes to explore (Open List is empty)
    while open_list:
        # Get the node with the lowest f-score (g + h) from the priority queue.
        # This is the "Best-First" decision: picking the most promising node.
        current_f, current_g, current = heapq.heappop(open_list)
        
        # Increment counter (used to demonstrate efficiency in your presentation)
        nodes_expanded += 1
        
        # Skip if we already processed this node (avoids cycles)
        if current in closed_set: 
            continue
        
        # --- GOAL CHECK ---
        # If we reached the target, stop and return result.
        if current == goal:
            return reconstruct_path(came_from, current), nodes_expanded
        
        # Mark current node as processed (move from Open to Closed)
        closed_set.add(current)
        
        # --- NEIGHBOR EXPANSION ---
        # Check Up, Down, Left, Right (4-directional movement).
        # This matches the Manhattan distance logic (no diagonals).
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            # Calculate neighbor coordinates
            neighbor = (current[0] + dx, current[1] + dy)
            
            # --- BOUNDARY & OBSTACLE CHECK ---
            # 1. Is neighbor inside the map? (0 <= neighbor < rows/cols)
            # 2. Is neighbor NOT an obstacle? (grid value == 0)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0]][neighbor[1]] == 0:
                
                # Calculate tentative cost to reach this neighbor (current cost + 1 step).
                # In grid search, each step usually costs 1.
                # NOTE: In Zhuang et al. (Paper 2), this '1' would be replaced by Energy Cost.
                tentative_g = current_g + 1
                
                # If this path to neighbor is better than any previous path found...
                # (Either we haven't seen it, or we found a cheaper way to get there)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Record this new best path
                    g_score[neighbor] = tentative_g
                    
                    # Calculate Heuristic (h-score) based on user selection (Manhattan vs. Euclidean)
                    # NOTE: In Wang et al. (Paper 1), this function dynamically switches between these two.
                    h = heuristic(neighbor, goal, heuristic_type)
                    
                    # --- STANDARD A* EQUATION ---
                    # f(n) = g(n) + h(n)
                    # This determines the priority in the Best-First Queue.
                    f_score = tentative_g + h
                    
                    # Add neighbor to Priority Queue to be explored later.
                    # The heap automatically sorts this based on f_score.
                    heapq.heappush(open_list, (f_score, tentative_g, neighbor))
                    
                    # Record parent for path reconstruction.
                    came_from[neighbor] = current
    
    # If loop finishes without finding goal, no path exists.
    return None, nodes_expanded

# ==============================================================================
# HEURISTIC FUNCTIONS (h(n))
# ==============================================================================
def heuristic(a, b, type):
    """
    Calculates the estimated cost from node 'a' to goal 'b'.
    This is the 'informed' part of A* that guides the search direction.
    
    Args:
        a (tuple): Current node coordinates
        b (tuple): Goal node coordinates
        type (str): 'Manhattan' or 'Euclidean'
    """
    # --- MANHATTAN DISTANCE ---
    # Formula: |x1 - x2| + |y1 - y2|
    # Best for grid movement (only Up/Down/Left/Right).
    # Admissible for 4-directional grids (never overestimates actual cost).
    # Because it matches the movement rules, it is more 'informed' than Euclidean here.
    if type == "Manhattan": 
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # --- EUCLIDEAN DISTANCE ---
    # Formula: sqrt((x1 - x2)^2 + (y1 - y2)^2)
    # Straight-line distance (as the crow flies).
    # Also admissible for 4-directional, but underestimates MORE than Manhattan.
    # This causes A* to explore more nodes to prove optimality.
    else: 
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

# ==============================================================================
# PATH RECONSTRUCTION
# ==============================================================================
def reconstruct_path(came_from, current):
    """
    Backtracks from the goal to the start using parent pointers.
    
    Args:
        came_from (dict): Map of node -> parent node
        current (tuple): The goal node
    
    Returns:
        list: Ordered path from Start -> Goal
    """
    # Start from the goal and work backwards to start using parent pointers
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    # Reverse the list so it goes Start -> Goal
    return total_path[::-1]

# ==============================================================================
# CONTROLLER FUNCTION (Called by UI)
# ==============================================================================
def run_astar(heuristic_type, obstacle_density):
    """
    Main controller function that generates the map and runs the search.
    Implements deterministic map generation for reproducible scientific results.
    """
    # --- 1. SET SEED BASED ON DENSITY ---
    # CRITICAL FOR REPRODUCIBILITY:
    # This ensures Density=20 always generates the EXACT same map every time.
    # This allows us to isolate the variable (Heuristic) rather than luck (Map Layout).
    np.random.seed(int(obstacle_density * 100))
    
    max_attempts = 5  # Safety net: Try to generate a solvable map up to 5 times
    grid = None
    path = None
    nodes = 0
    
    # --- 2. RETRY LOOP (Deterministic because seed is fixed above) ---
    # Random maps can accidentally block the start from the goal.
    # This loop ensures we don't show a "Failed" demo to the class.
    for attempt in range(max_attempts):
        # Generate Empty Map (25x25 grid)
        grid = [[0 for _ in range(25)] for _ in range(25)]
        
        # Add Random Obstacles based on slider density
        for i in range(25):
            for j in range(25):
                # np.random.random() gives 0.0 to 1.0. If less than density%, make it a wall (1)
                if np.random.random() < obstacle_density/100:
                    grid[i][j] = 1
        
        # Ensure Start (0,0) and Goal (24,24) are never blocked
        grid[0][0] = 0
        grid[24][24] = 0
        
        start = (0, 0)
        goal = (24, 24)
        
        # Run A* Algorithm
        path, nodes = a_star_search(grid, start, goal, heuristic_type)
        
        # If path found, break the retry loop
        if path:
            break
    
    # --- 3. VISUALIZATION ---
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
        # 'Nodes Expanded' is the key metric for heuristic efficiency
        result = f"✅ Path Found!\nDensity: {obstacle_density}%\nNodes Expanded: {nodes}\nPath Length: {len(path)}"
    else:
        # Prepare failure message (rare due to retry loop)
        result = f"❌ No Path Found\nDensity: {obstacle_density}%\n(try different density)"
    
    # Add legend and grid lines
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    
    # Return the plot image and the text result to the UI
    return fig, result

# ==============================================================================
# GRADIO USER INTERFACE
# ==============================================================================
# Creates a custom layout block for the web app
with gr.Blocks(title="A* Heuristic Lab", css=".gradio-container {max-width: 600px !important;}") as demo:
    gr.Markdown("# 🧠 A* Heuristic Lab (Controlled Experiment)")
    gr.Markdown("Compare Manhattan vs. Euclidean on the **same map**")
    
    with gr.Row():
        with gr.Column():
            # Input 1: Dropdown to choose Heuristic
            # This is the independent variable in our experiment
            heuristic_dropdown = gr.Dropdown(
                choices=["Manhattan", "Euclidean"],
                value="Manhattan",
                label="Heuristic Function (h(n))"
            )
            # Input 2: Slider to choose Obstacle Density
            # Fixed steps (10, 15, 20, 25) ensure reproducible seeds
            density_slider = gr.Slider(
                minimum=10, maximum=25, value=15, step=5,
                label="Obstacle Density (%)"
            )
            # Input 3: Button to trigger the search
            run_button = gr.Button("Run A* Search", variant="primary")
        
        with gr.Column():
            # Output 1: The Matplotlib Plot
            output_plot = gr.Plot(label="Path Visualization")
            # Output 2: The Text Results (Nodes Expanded, Path Length)
            output_text = gr.Textbox(label="Results")
    
    # --- EVENT LISTENER ---
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