import streamlit as st
import heapq
import matplotlib.pyplot as plt
import numpy as np

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
        if current in closed_set: continue
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
    if type == "Manhattan": return abs(a[0] - b[0]) + abs(a[1] - b[1])
    else: return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

# --- 2. Streamlit App Layout ---
st.set_page_config(page_title="A* Rescue Game", layout="wide")
st.title("🤖 A* Dynamic Rescue Challenge")
st.markdown("Click buttons to add obstacles. Watch A* replan!")

# Initialize Session State for Grid
if 'grid' not in st.session_state:
    st.session_state.grid = [[0 for _ in range(15)] for _ in range(15)]
    st.session_state.start = (0, 0)
    st.session_state.goal = (14, 14)

# Sidebar Controls
st.sidebar.header("Controls")
heuristic_type = st.sidebar.selectbox("Heuristic", ["Manhattan", "Euclidean"])
clear_btn = st.sidebar.button("Clear Obstacles")
run_btn = st.sidebar.button("Run A* Pathfinding", type="primary")

# Handle Clear
if clear_btn:
    st.session_state.grid = [[0 for _ in range(15)] for _ in range(15)]
    st.rerun()

# Display Grid Info
st.subheader("Mission Map")
grid_array = np.array(st.session_state.grid)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(grid_array, cmap='binary')
ax.plot(st.session_state.start[1], st.session_state.start[0], 'go', markersize=15, label='Start')
ax.plot(st.session_state.goal[1], st.session_state.goal[0], 'bo', markersize=15, label='Goal')
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.info("💡 Click 'Add Random Obstacles' to challenge the robot.")

if st.button("🎲 Add Random Obstacles"):
    for _ in range(5):
        r, c = np.random.randint(0, 15), np.random.randint(0, 15)
        if (r,c) != st.session_state.start and (r,c) != st.session_state.goal:
            st.session_state.grid[r][c] = 1
    st.rerun()

# Run Algorithm
if run_btn:
    path, nodes = a_star_search(st.session_state.grid, st.session_state.start, st.session_state.goal, heuristic_type)
    st.subheader("Mission Result")
    if path:
        st.success(f"✅ Path Found! Expanded {nodes} nodes.")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.imshow(np.array(st.session_state.grid), cmap='binary')
        path_y, path_x = zip(*path)
        ax2.plot(path_x, path_y, 'r-', linewidth=3, label='A* Path')
        ax2.plot(st.session_state.start[1], st.session_state.start[0], 'go', markersize=15)
        ax2.plot(st.session_state.goal[1], st.session_state.goal[0], 'bo', markersize=15)
        ax2.legend()
        st.pyplot(fig2)
        
        st.markdown("""
        ### 🧠 AI Insight:
        - **Nodes Expanded:** Represents **Computation Time** (Paper 1: Wang et al.).
        - **Path Shape:** Represents **Smoothness/Energy** (Paper 2: Zhuang et al.).
        """)
    else:
        st.error("❌ No Path Found! The robot is trapped.")