from func.datastruct import Point, Machine
import matplotlib.pyplot as plt

# --- Current Machine Data --- #

machines = [
    Machine(id=1, shape='rectangle', width=4.0, height=3.0, 
            access_point=Point(1.5, 0),
            position=Point(0, 0)),
    Machine(id=2, shape='l_shape', width=5.0, height=4.0, 
            access_point=Point(-1.0, 1.0),
            l_cutout_width=2.0, l_cutout_height=2.0,
            position=Point(6, 0)),
    Machine(id=3, shape='rectangle', width=3.5, height=2.5, 
            access_point=Point(0, -1.0),
            position=Point(13, 0)),
    Machine(id=4, shape='l_shape', width=4.5, height=3.5, 
            access_point=Point(1.0, -0.5),
            l_cutout_width=1.5, l_cutout_height=1.5,
            position=Point(18, 0)),
    Machine(id=5, shape='l_shape', width=3, height=5.5, 
            access_point=Point(1.0, -1.0),
            l_cutout_width=1, l_cutout_height=2,
            position=Point(0, -8)),
    Machine(id=6, shape='l_shape', width=5, height=3, 
            access_point=Point(1.0, -1.0),
            l_cutout_width=2, l_cutout_height=2,
            position=Point(5, -8)),
    Machine(id=7, shape='l_shape', width=3.5, height=2.5, 
            access_point=Point(1.0, -1.0),
            l_cutout_width=1, l_cutout_height=2,
            position=Point(10, -8)),
    Machine(id=8, shape='rectangle', width=3.5, height=2.5, 
            access_point=Point(0, 0),
            position=Point(15, -8)),
]

# --- Plot setup --- #
fig, ax = plt.subplots(figsize=(14, 10))

for machine in machines:
    color = 'skyblue' if machine.shape == 'rectangle' else 'lightgreen'
    corners = machine.get_corners()
    
    xs = [p.x for p in corners] + [corners[0].x]
    ys = [p.y for p in corners] + [corners[0].y]
    
    ax.fill(xs, ys, color=color, alpha=0.5, edgecolor='black', linewidth=1.5)
    
    # Mark access point
    ap = machine.get_access_point_world()
    ax.plot(ap.x, ap.y, 'ro', markersize=8)
    ax.text(ap.x + 0.3, ap.y + 0.3, f"M{machine.id}", fontsize=9, color='red', 
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add machine ID at center
    center_x = sum(p.x for p in corners) / len(corners)
    center_y = sum(p.y for p in corners) / len(corners)
    ax.text(center_x, center_y, f"ID {machine.id}", fontsize=10, 
            ha='center', va='center', fontweight='bold')

# --- Plot styling --- #
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel("X (meters)", fontsize=12)
ax.set_ylabel("Y (meters)", fontsize=12)
ax.set_title("Machine Layout: 8 Machines (Rectangles and L-Shapes)", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Create custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='skyblue', alpha=0.5, edgecolor='black', label='Rectangle'),
    Patch(facecolor='lightgreen', alpha=0.5, edgecolor='black', label='L-Shape'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='Access Point')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()