import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import shapely.affinity

class MultiSensoryTile:
    def __init__(self, tile_size: float, sensor_number: int) -> None:
        self.TS = tile_size
        self.NS = sensor_number
        self.polygon = Polygon([(0, 0), (self.TS, 0), (self.TS, self.TS), (0, self.TS)])
        self.sensors = self._create_sensors()
        self.matrix_position = None

    def _create_sensors(self):
        """Create sensors distributed evenly across the tile."""
        sensor_points = []
        step = self.TS / self.NS
        for i in range(self.NS):
            for j in range(self.NS):
                sensor = Point(step * (i + 0.5), step * (j + 0.5))
                sensor_points.append(sensor)
        return sensor_points

    @property
    def center(self) -> tuple:
        """Return the current center of the polygon."""
        center = self.polygon.centroid.coords.xy
        return np.array([center[0][0], center[1][0]])

    @property
    def sensors_positions(self) -> list:
        """Return the positions of the sensors as a list of numpy arrays."""
        return [np.array([sensor.x, sensor.y]) for sensor in self.sensors]

    @center.setter
    def center(self, new_center: tuple) -> None:
        """Translate the tile and its sensors to a new center."""
        translation_x, translation_y = np.subtract(new_center, self.center)
        # Translate the polygon
        self.polygon = shapely.affinity.translate(self.polygon, xoff=translation_x, yoff=translation_y, zoff=0.0)
        # Translate the sensors
        new_positions = []
        for sensor in self.sensors:
            sensor = shapely.affinity.translate(sensor, xoff=translation_x, yoff=translation_y, zoff=0.0)
            new_positions.append(sensor)
        self.sensors = new_positions

    def __str__(self) -> str:
        """Return a string representation of the tile's center and matrix position."""
        return f"Center: {self.center}, Matrix Position: {self.matrix_position}"

    def plot(self, show=False) -> None:
        """Plot the tile and its sensors."""
        x, y = self.polygon.exterior.xy
        plt.plot(x, y, color='black')
        # Plot the sensors
        for sensor in self.sensors:
            plt.scatter(sensor.x, sensor.y, color='red', marker='x')
        # Ensure aspect ratio is equal
        plt.gca().set_aspect('equal', adjustable='box')
        if show:
            plt.show()

if __name__ == '__main__':
    # Example usage:
    tile = MultiSensoryTile(10, 1)
    print(tile)  # Initial position
    tile.center = (0, 0)  # Move the tile
    tile.plot(show=True)  # Plot the tile and sensors
