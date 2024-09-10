from shapely.geometry import Point, Polygon
import shapely.affinity
import numpy as np
from matplotlib import pyplot as plt


class MultiSensoryTile:
    def __init__(self, tile_size:float = 1, sensor_number = 1) -> None:
        self.TS = tile_size
        self.NS = sensor_number
        self.polygon = Polygon([(0, 0), (self.TS, 0), (self.TS, self.TS), (0, self.TS)])
        self.sensors = self._create_sensors()
        self.matrix_position = None

    def _create_sensors(self):
        sensor_points = []
        for i in range(self.NS):
            for j in range(self.NS):
                sensor = Point(self.TS/self.NS * (i+self.TS/2), self.TS/self.NS * (j+self.TS/2))
                sensor_points.append(sensor)
        return sensor_points
    
    @property
    def center(self) -> tuple:
        center = self.polygon.centroid.coords.xy
        return np.array([center[0][0], center[1][0]])
    
    @property
    def sensors_positions(self) -> list:
        return [np.array([sensor.x, sensor.y]) for sensor in self.sensors]

    @center.setter
    def center(self, new_center: tuple) -> None:
        translation_x, translation_y = np.subtract(new_center, self.center)
        self.polygon = shapely.affinity.translate(self.polygon, xoff=translation_x, yoff=translation_y, zoff=0.0)
        new_positions = []
        for sensor in self.sensors:
            sensor = shapely.affinity.translate(sensor, xoff=translation_x, yoff=translation_y, zoff=0.0)
            new_positions.append(sensor)
        self.sensors = new_positions

    def __str__(self) -> str:
        return (self.center, self.matrix_position)

    def plot(self):
        x, y = self.polygon.exterior.xy
        plt.plot(x, y, color='black')
        for sensor in self.sensors:
            plt.scatter(sensor.x, sensor.y, color='black', marker='x')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        
if __name__ == '__main__':
    pass