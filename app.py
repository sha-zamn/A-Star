import streamlit as st
import heapq
import matplotlib.pyplot as plt
import random

# --- 1. A* Algorithm (Standard) ---
def a_star_search(grid, start, goal, heuristic_type):
    rows, cols = len(grid), len(grid[0])
    open_list = []
    heapq.heappush(open_list, (0, 0, start))
    came_from = {}
    g_score = {start: 0}
    closed_set = set()
    nodes_expanded = 0

    while open_list:
        current_f, current_g, current = heapq.heappop(open_list)
        nodes_expanded += 1

        if current in closed_set:
            continue
        if current == goal:
            return reconstruct_path(came_from, current), nodes_expanded

        closed_set.add(current)

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0]][neighbor[1]] == 0:
                tentative_g = current_g + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    h = heuristic(neighbor, goal, heuristic_type)
                    f_score = tentative_g + h
                    heapq.heappush(open_list, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current
    return None, nodes_expanded

def heuristic(a, b, type):
    if type == "Manhattan":
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    else:  # Euclidean
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

# --- 2. Streamlit App Layout ---
st.set_page_config(page_title="A* Search Demo", layout="centered")

st.title("🤖 Interactive A* Search")
st.markdown("Scan this on your phone! Adjust settings to see how A* behaves.")

# Sidebar Controls (Mobile Friendly)
st.sidebar.header("Controls")
heuristic_type = st.sidebar.selectbox("Heuristic Function", ["Manhattan", "Euclidean"])
obstacle_density = st.sidebar.slider("Obstacle Density (%)", 10, 50, 20)
run_button = st.sidebar.button("Run A* Search", type="primary")

# Generate Map
@st.cache_data
def generate_map(rows, cols, density):
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if random.random() < (density / 100.0):
                grid[r][c] = 1
    # Ensure start and goal are clear
    grid[0][0] = 0
    grid[rows-1][cols-1] = 0
    return grid

rows, cols = 15, 15
grid = generate_map(rows, cols, obstacle_density)
start_pos = (0, 0)
goal_pos = (rows-1, cols-1)

# Run Algorithm
if run_button:
    path, nodes = a_star_search(grid, start_pos, goal_pos, heuristic_type)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap='binary')
    
    if path:
        path_y, path_x = zip(*path)
        ax.plot(path_x, path_y, color='red', linewidth=2, label='Path')
        ax.plot(start_pos[1], start_pos[0], 'go', markersize=10, label='Start')
        ax.plot(goal_pos[1], goal_pos[0], 'bo', markersize=10, label='Goal')
        ax.set_title(f"Nodes Expanded: {nodes}")
    else:
        ax.set_title("No Path Found!")
    
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    st.success(f"✅ Path Found! Expanded {nodes} nodes.")
else:
    st.info("👈 Adjust settings in the sidebar and click 'Run'")