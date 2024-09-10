from MSTile import MultiSensoryTile
from WTetromino import WTetromino
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from DensityMetrics import DensityMetrics

class ContactBoard:
    def __init__(self, board_shape) -> None:
        self.board_shape = board_shape
        self.tiles = self._create_tiles()
        self.contour = self._create_contour()

    def _create_contour(self) -> np.array:
        contour_vertices = [(0,0), (0, self.board_shape[0]), (self.board_shape[1], self.board_shape[0]), (self.board_shape[1], 0)]
        contour_vertices = [(point[0] - 0.5, point[1] - 0.5) for point in contour_vertices]
        contour = Polygon(contour_vertices)
        return contour
    
    def _create_tiles(self) -> None:
        tiles = []
        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1]):
                tile = MultiSensoryTile(1)
                tile.center = np.array([j, i])
                tile.matrix_position = np.array([j, i])
                tiles.append(tile)
        return tiles

    def has_tetromino_inside(self, tetromino: WTetromino) -> bool:
            return self.contour.contains(tetromino.polygon)
    
    def has_point_inside(self, point: tuple) -> bool:
        return self.contour.contains(Point(point))
    
    def get_contact_mask(self, tetromino: WTetromino) -> np.array:
        contact_mask = np.zeros(self.board_shape, dtype=int)
        for tile in self.tiles:
            if tetromino.polygon.buffer(1e-6).contains(tile.sensor):
                row_index = self.board_shape[0] - tile.matrix_position[1] - 1
                col_index = tile.matrix_position[0]
                contact_mask[row_index, col_index] = 1
        return contact_mask

    def plot(self, tetromino: WTetromino = None, plot_density = True) -> None:
        for tile in self.tiles:
            tile.plot()
        #plot contour
        x, y = self.contour.exterior.xy
        #plt.plot(x, y, color='red', linestyle='dashed')
        if tetromino is not None:
            tetromino.plot()


        if tetromino and plot_density:
            DENSITY_CMAP = 'inferno'#'viridis'

            ax = plt.subplot()
            circles_positions = tetromino.mass_points            
            
            Hmax,Wmax = self.contour.exterior.xy[1][0], self.contour.exterior.xy[0][0]
            Hmin,Wmin = self.contour.exterior.xy[1][2], self.contour.exterior.xy[0][2]

            resolution = 50
            x, y = np.meshgrid(np.linspace(Hmin, Hmax, resolution), np.linspace(Wmin, Wmax, resolution))
            points = np.vstack([x.ravel(), y.ravel()])
            densities = np.array([DensityMetrics.exponential((xi, yi), circles_positions) for xi, yi in points.T])
            densities = (densities - np.min(densities)) / (np.max(densities) - np.min(densities)) #Normalize to [0, 1]
            densities = densities.reshape(resolution, resolution)

            ax.imshow(densities, extent=[Hmin, Hmax, Wmin, Wmax], origin='lower', cmap=DENSITY_CMAP)
            ax.set_aspect('equal')


        #remove axis
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable='box')
        #save
        plt.savefig('all.png', dpi=600, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    cb = ContactBoard([4,4])
    tetromino = WTetromino(shape='T', scaler=1)
    tetromino.center = np.array([1.5, 1.5])
    tetromino.rotate(-45)
    cb.plot(tetromino=tetromino)
    cm = cb.get_contact_mask(tetromino)
    print(cm)
    pass