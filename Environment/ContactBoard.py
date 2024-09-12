from MSTile import MultiSensoryTile
from WTetromino import WTetromino
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from DensityMetrics import DensityMetrics

class ContactBoard:
    def __init__(self, board_shape: list, tile_size: float, center: tuple = (0, 0)) -> None:
        self.tile_size = tile_size
        self.board_shape = board_shape
        self.center = np.array(center)
        self.tiles = self._create_tiles()
        self.contour = self._create_contour()

    def _create_contour(self) -> Polygon:
        """Create the contour of the board as a polygon based on the board's center."""
        half_width = self.board_shape[1] * self.tile_size / 2
        half_height = self.board_shape[0] * self.tile_size / 2
        contour_vertices = [
            (self.center[0] - half_width, self.center[1] - half_height),
            (self.center[0] - half_width, self.center[1] + half_height),
            (self.center[0] + half_width, self.center[1] + half_height),
            (self.center[0] + half_width, self.center[1] - half_height)
        ]
        contour = Polygon(contour_vertices)
        return contour

    def _create_tiles(self) -> list:
        """Create tiles and arrange them in a grid pattern based on the board's center."""
        tiles = []
        half_width = self.board_shape[1] * self.tile_size / 2
        half_height = self.board_shape[0] * self.tile_size / 2
        for i in range(self.board_shape[0]):  # Loop over rows
            for j in range(self.board_shape[1]):  # Loop over columns
                tile = MultiSensoryTile(tile_size=self.tile_size, sensor_number=1)
                # Set the tile's center based on the board's center and tile size
                tile_center_x = self.center[0] - half_width + (j + 0.5) * self.tile_size
                tile_center_y = self.center[1] - half_height + (i + 0.5) * self.tile_size
                tile.center = np.array([tile_center_x, tile_center_y])
                tile.matrix_position = np.array([j, i])
                tiles.append(tile)
        return tiles

    @property
    def center(self) -> tuple:
        return self._center

    @center.setter
    def center(self, new_center: tuple) -> None:
        """Update the board center and recompute the tiles and contour."""
        self._center = np.array(new_center)
        self.tiles = self._create_tiles()  # Recreate tiles with new center
        self.contour = self._create_contour()  # Recreate contour with new center

    @property
    def sensor_positions(self) -> list:
        X = [sensor.x for tile in self.tiles for sensor in tile.sensors]
        Y = [sensor.y for tile in self.tiles for sensor in tile.sensors]
        X = np.array(X, dtype=float).reshape(self.board_shape)
        Y = np.array(Y, dtype=float).reshape(self.board_shape)
        return X,Y


    def has_tetromino_inside(self, tetromino: WTetromino) -> bool:
            return self.contour.contains(tetromino.polygon)
    
    def has_point_inside(self, point: tuple) -> bool:
        return self.contour.contains(Point(point))
    
    def get_contact_mask(self, tetromino: WTetromino) -> np.array:
        contact_mask = np.zeros(self.board_shape, dtype=int)
        for tile in self.tiles:
            if any(tetromino.polygon.buffer(1e-6).contains(sensor) for sensor in tile.sensors):
                row_index = self.board_shape[0] - tile.matrix_position[1] - 1
                col_index = tile.matrix_position[0]
                contact_mask[row_index, col_index] = 1
        return contact_mask

    def plot(self, tetromino: WTetromino = None, plot_density = False) -> None:
        for tile in self.tiles:
            tile.plot()
        #plot contour
        x, y = self.contour.exterior.xy
        plt.plot(x, y, color='red', linestyle='dashed')
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
        #plt.xticks([])
        #plt.yticks([])
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable='box')
        #save
        plt.savefig('all.png', dpi=600, bbox_inches='tight')
        #plt.show()


if __name__ == '__main__':
    pass