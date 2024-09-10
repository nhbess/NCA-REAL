
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import matplotlib.pyplot as plt
import random
from Shapes import tetros_dict
import numpy as np
from scipy.stats import gaussian_kde
from DensityMetrics import DensityMetrics

def generate_points(shape:str = 'T', scaler:int = 1, save_fig:bool=False, file_name:str='test.png'):
    g = cfg.Geometry()

    polygon_points = tetros_dict[shape]*scaler
    polygon_points = [[point[0], point[1]] for point in polygon_points]


    for point in polygon_points:
        g.point(point)


    POINTS = [10,20]
    POINTS = [5,25]
    POINTS = [5,10]

    def random_nodes():
        return random.randint(*POINTS)
    for i in range(len(polygon_points)-1):
        g.spline([i,i+1], el_on_curve=random_nodes(), el_distrib_val=1)
    g.spline([len(polygon_points)-1,0], el_on_curve=random_nodes(), el_distrib_val=1)

    g.surface(range(len(polygon_points)))
    mesh = cfm.GmshMesh(g)

    mesh.el_type = 2    # 3 is for triangles, 2 is for quadrilaterals 
    mesh.el_size_factor = 1#0.05

    coords, edof, dofs, bdofs, elementmarkers = mesh.create()

    if save_fig:
        cfv.figure()
        cfv.draw_mesh(coords=coords, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)
        for i in range(len(coords)):
            plt.scatter(coords[i][0], coords[i][1], c='black')

        #remove axis
        plt.xticks([])
        plt.yticks([])

        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.clf()

    return coords

def plot_density(circles_positions):
    DENSITY_CMAP = 'inferno'#'viridis'
    #DENSITY_CMAP = 'viridis'

    ax = plt.subplot()

    #PLOT CIRCLES
    X = np.array([p[0] for p in circles_positions])
    Y = np.array([p[1] for p in circles_positions])
    ax.scatter(X, Y, c='w', alpha=0.5, s=20, marker='o')
    
    #PLOT DENSITY
    H,W = 5,5

    resolution = 50
    x, y = np.meshgrid(np.linspace(0, H, resolution), np.linspace(0, W, resolution))
    points = np.vstack([x.ravel(), y.ravel()])
    densities = np.array([DensityMetrics.exponential((xi, yi), circles_positions) for xi, yi in points.T])
    densities = (densities - np.min(densities)) / (np.max(densities) - np.min(densities)) #Normalize to [0, 1]
    densities = densities.reshape(resolution, resolution)

    ax.imshow(densities, extent=[0, H, 0, W], origin='lower', cmap=DENSITY_CMAP)
    ax.set_aspect('equal')

    sensors_positions = []
    N = 4
    for i in range(N):
        for j in range(N):
            sensors_positions.append([H/N*i + 0.5, W/N*j+ 0.5])

    #scater sensors
    X = np.array([p[0] for p in sensors_positions])
    Y = np.array([p[1] for p in sensors_positions])
    ax.scatter(X, Y, c='lightgreen', alpha=0.5, marker='x')

    plt.tight_layout()

    plt.xlim([0, H])
    plt.ylim([0, W])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(f'density.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()

if __name__ == '__main__': 
    points = generate_points(shape = 'L', save_fig=True, file_name='test.png')
    points = np.array(points) + np.array([1.5,1.5])
    plot_density(points)