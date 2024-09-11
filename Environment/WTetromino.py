import sys
import matplotlib.pyplot as plt
import numpy as np
import shapely.affinity
from shapely.geometry import Polygon
import Shapes
import matplotlib.pyplot as plt
from PointsByMesh import generate_points


class WTetromino:
    def __init__(self, shape:str, scaler:float = 1) -> None:
        self.__constructor_vertices = Shapes.tetros_dict[shape]
        self.__polygon = Polygon(self.__constructor_vertices*scaler)
        self.__angle = 0.0
        self.__mass_points = Polygon(generate_points(shape, scaler))

    @property
    def polygon(self) -> Polygon:
        return self.__polygon
    @property
    def center(self) -> tuple:
        center = self.__polygon.centroid.coords.xy
        return np.array([center[0][0], center[1][0]])

    @property
    def angle(self) -> float:
        return self.__angle
    
    @property
    def vertices(self) -> np.array:
        vertices = self.__polygon.exterior.coords.xy
        return np.array([vertices[0], vertices[1]]).T

    @property
    def constructor_vertices(self) -> np.array:
        return self.__constructor_vertices.tolist()

    @property
    def mass_points(self) -> np.array:
        mp_vertices = self.__mass_points.exterior.coords.xy
        return np.array([mp_vertices[0], mp_vertices[1]]).T
    

    @center.setter
    def center(self, new_center: tuple) -> None:
        old_center = self.center
        self.__polygon = shapely.affinity.translate(self.__polygon, xoff=new_center[0] - self.center[0], yoff=new_center[1] - self.center[1], zoff=0.0)
        self.__mass_points = shapely.affinity.translate(self.__mass_points, xoff=new_center[0] - old_center[0], yoff=new_center[1] - old_center[1], zoff=0.0)
    
    @angle.setter
    def angle(self, angle: float) -> None:
        self.rotate(angle - self.__angle)
    
    def rotate(self, angle: float) -> None:
        old_center = self.__polygon.centroid.xy[0][0], self.__polygon.centroid.xy[1][0]
        self.__polygon = shapely.affinity.rotate(self.__polygon, angle, origin=old_center, use_radians=False)
        self.__mass_points = shapely.affinity.rotate(self.__mass_points, angle, origin=old_center, use_radians=False)
        self.__angle = (self.__angle + angle) % 360

    def translate(self, direction) -> None:
        self.__polygon = shapely.affinity.translate(self.__polygon, xoff=direction[0], yoff=direction[1], zoff=0.0)
        self.__mass_points = shapely.affinity.translate(self.__mass_points, xoff=direction[0], yoff=direction[1], zoff=0.0)

    def plot(self, show = False) -> None:
        x_values, y_values = zip(*self.vertices)
        plt.plot(x_values, y_values)  # Plot the vertices
        plt.plot(self.center[0],self.center[1], 'ro')  # Mark the first vertex with a red dot
        #plt.scatter(self.mass_points[:,0], self.mass_points[:,1], c='black', alpha=0.5) 
        plt.gca().set_aspect('equal', adjustable='box')
        if show: plt.show()
    
    
    def print_info(self) -> None:
        print('center: {}'.format(self.center))
        print('angle: {}'.format(self.__angle))
 
def test():
    SCALE = 15
    tetromino = WTetromino(shape='L', scaler=SCALE)
    tetromino.center = np.array([100, 0])
    tetromino.angle = 45
    tetromino.plot()

    tetromino.rotate(-45)
    tetromino.translate(np.array([0, 100]))
    tetromino.plot()

if __name__ == '__main__':
    pass