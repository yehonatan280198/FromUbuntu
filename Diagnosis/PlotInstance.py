import ast
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, RegularPolygon

def create_map(map_name):
    file_path = f"OurResearch.domain/{map_name}.map"
    with open(file_path, "r") as file:
        lines = file.readlines()

    map_start_index = lines.index("map\n") + 1
    map_lines = lines[map_start_index:]

    currMap = []
    rows, cols, numOfFreeCells, numOfObs = 0, 0, 0, 0

    for line in map_lines:
        cols = len(line.strip())
        rows += 1
        for char in line.strip():
            if char == ".":
                currMap.append(0)
                numOfFreeCells += 1
            else:
                currMap.append(1)
                numOfObs += 1

    return {
        "Rows": rows,
        "Cols": cols,
        "Map": currMap,
        "ObsRatio": numOfObs / (rows * cols),
        "FreeCells": numOfFreeCells
    }

def read_locs_from_file(num_of_instance):
    file_agents_name = f"Agent_Goal_locations_files/Agents_Scale/{mapName.split('.')[0]}_Map_Agent_Locs_instance_{num_of_instance - 1}.txt"
    file_goals_name = f"Agent_Goal_locations_files/Agents_Scale/{mapName.split('.')[0]}_Map_Goal_Locs_instance_{num_of_instance - 1}.txt"

    with open(file_agents_name, "r") as f:
        Agents_Positions = [
            ast.literal_eval(line.strip())
            for _, line in zip(range(num_of_agents), f)
        ]

    with open(file_goals_name, "r") as f:
        Goals_Locations = [
            ast.literal_eval(line.strip())
            for _, line in zip(range(num_of_goals), f)
        ]

    return Agents_Positions, Goals_Locations

def draw_instance(map_name, num_of_instance, selected_agent=None):
    map_data = create_map(map_name)
    rows = map_data["Rows"]
    cols = map_data["Cols"]
    grid = map_data["Map"]

    Agents_Positions, Goals_Locations = read_locs_from_file(num_of_instance)

    fig, ax = plt.subplots(figsize=(10, 10))

    # ציור הגריד
    for r in range(rows):
        for c in range(cols):
            val = grid[r * cols + c]
            facecolor = "white" if val == 0 else "black"

            rect = Rectangle(
                (c, r), 1, 1,
                facecolor=facecolor,
                edgecolor="gray",
                linewidth=0.5
            )
            ax.add_patch(rect)

    # ציור מטרות - משולש ירוק
    for loc in Goals_Locations:
        gr, gc = divmod(loc, cols)
        triangle = RegularPolygon(
            (gc + 0.5, gr + 0.5),
            numVertices=3,
            radius=0.28,
            orientation=0,
            facecolor="green",
            edgecolor="darkgreen",
            linewidth=1
        )
        ax.add_patch(triangle)

    # ציור סוכנים
    selected_pos = None

    for i, loc in enumerate(Agents_Positions):
        ar, ac = divmod(loc, cols)

        if selected_agent is not None and i == selected_agent:
            facecolor = "orange"
            edgecolor = "darkorange"
            selected_pos = (ac, ar)  # x, y
        else:
            facecolor = "blue"
            edgecolor = "navy"

        circle = Circle(
            (ac + 0.5, ar + 0.5),
            radius=0.28,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1
        )
        ax.add_patch(circle)

    # כותרת
    title = f"Map: {map_name}, instance: {num_of_instance}"
    if selected_agent is None:
        title += " | selected agent: None"
    else:
        x, y = selected_pos
        title += f" | selected agent: {selected_agent + 1} at (x={x}, y={y})"

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)  # כדי ש-(0,0) יהיה למעלה
    ax.set_aspect("equal")
    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.grid(False)
    ax.set_title(title)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.show()

def draw_all_agent_choices(map_name, num_of_instance):
    Agents_Positions, _ = read_locs_from_file(num_of_instance)

    # קודם ציור בלי סוכן נבחר
    draw_instance(map_name, num_of_instance, selected_agent=None)

    # אחר כך כל פעם סוכן אחר
    for agent_idx in range(len(Agents_Positions)):
        draw_instance(map_name, num_of_instance, selected_agent=agent_idx)

# =========================
# הגדרות
num_of_agents = 25
num_of_goals = 100
mapName = "maze-32-32-2"

# הרצה
draw_all_agent_choices(mapName, num_of_instance=1)