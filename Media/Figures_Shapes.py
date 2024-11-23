from matplotlib import pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Function to create a circle
def _create_circle(n_points=50, radius=1):
    angles = np.linspace(0, 2 * np.pi, n_points + 1)[:-1]
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    return np.array([xs, ys]).T

# Scaling factor
scaler = 25

# Define the shapes
tetros_dict = {
    'A': _create_circle(n_points=50, radius=scaler),
    'O': np.array([(0, 0), (0, 1), (1, 1), (1, 0)]) * 72,
    'L': np.array([(0, 0), (0, 3), (1, 3), (1, 1), (2, 1), (2, 0)]) * scaler,
    'S': np.array([(0, 0), (0, 1), (1, 1), (1, 2), (3, 2), (3, 1), (2, 1), (2, 0)]) * scaler,
    'T': np.array([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 1), (3, 1), (3, 0)]) * scaler,
    'Z': np.array([(0, 2), (2, 2), (2, 1), (3, 1), (3, 0), (1, 0), (1, 1), (0, 1)]) * scaler,
    'J': np.array([(0, 0), (0, 1), (1, 1), (1, 3), (2, 3), (2, 0)]) * scaler,
    'H': _create_circle(n_points=6, radius=scaler),
}

masses_dict = {
    'A': 1,
    'O': 4,
    'L': 4,
    'S': 4,
    'T': 4,
    'Z': 4,
    'J': 4,
    'H': 4,
}

# Create a color palette
def create_palette(num_colors):
    cmap = plt.cm.get_cmap("tab10")
    return [cmap(i) for i in range(num_colors)]

# Calculate the shortest edge for each shape
def calculate_shortest_edge(shape):
    distances = np.linalg.norm(np.diff(np.vstack([shape, shape[0]]), axis=0), axis=1)
    return np.min(distances)

if __name__ == '__main__':
    shapes = list(tetros_dict.values())
    names = list(tetros_dict.keys())
    palette = create_palette(len(shapes))

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(shapes):
            shape = shapes[i]
            name = names[i]

            # Align shapes to start from origin
            shape[:, 1] = shape[:, 1] - min(shape[:, 1])
            shape[:, 0] = shape[:, 0] - min(shape[:, 0])

            # Extract x and y for plotting
            xs, ys = shape[:, 0], shape[:, 1]
            xs = [*xs, xs[0]]
            ys = [*ys, ys[0]]

            # Plot the shape
            ax.plot(xs, ys, color="black", linewidth=1.5, zorder=2)
            ax.fill(xs, ys, color=palette[i], alpha=1, zorder=1)

            # Calculate and annotate the shortest edge
            shortest_edge = calculate_shortest_edge(shape)
            ax.text(
                0.5,
                -0.1,
                f"Shortest Edge: {shortest_edge:.1f}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="black",
            )

            # Set the title and axis settings
            ax.set_title(name, fontsize=14)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
        else:
            # Hide any unused subplots
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("shapes_with_shortest_edges.png", dpi=600, bbox_inches="tight", pad_inches=0, transparent=False)
    plt.show()
