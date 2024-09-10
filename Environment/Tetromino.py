import matplotlib.pyplot as plt
import numpy as np
import shapely.affinity
from shapely.geometry import Polygon
import Shapes

class Tetromino:
    def __init__(self, constructor_vertices:list[tuple], scaler:float = 1) -> None:
        self.__constructor_vertices = constructor_vertices
        self.id = id
        self.scaler = scaler
        self.polygon = Polygon(constructor_vertices*scaler)
        self.__angle = 0.0
    
    @property
    def center(self) -> tuple:
        center = self.polygon.centroid.coords.xy
        return np.array([center[0][0], center[1][0]])
    
    @property
    def vertices(self) -> np.array:
        vertices = self.polygon.exterior.coords.xy
        return np.array([vertices[0], vertices[1]]).T
    @property
    def constructor_vertices(self) -> np.array:
        return self.__constructor_vertices.tolist()
    @center.setter
    def center(self, new_center: tuple) -> None:
        self.polygon = shapely.affinity.translate(self.polygon, xoff=new_center[0] - self.center[0], yoff=new_center[1] - self.center[1], zoff=0.0)

    def rotate(self, angle: float) -> None:
        self.polygon = shapely.affinity.rotate(self.polygon, angle, origin='centroid', use_radians=False)
        self.__angle = (self.__angle + angle)%360

    def translate(self, direction) -> None:
        self.polygon = shapely.affinity.translate(self.polygon, xoff=direction[0], yoff=direction[1], zoff=0.0)
        
    def plot(self) -> None:
        x_values, y_values = zip(*self.vertices)
        plt.plot(x_values, y_values)  # Plot the vertices
        plt.plot(self.center[0],self.center[1], 'ro')  # Mark the first vertex with a red dot
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Tetromino')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
  
    def set_angle(self, new_angle):
        angle = new_angle - self.__angle
        self.polygon = shapely.affinity.rotate(self.polygon, angle, origin='centroid', use_radians=False)
        self.__angle = new_angle
    
    @property
    def angle(self) -> float:
        return self.__angle
    
    def print_info(self) -> None:
        print('center: {}'.format(self.center))
        print('angle: {}'.format(self.__angle))
 
def test():
    SCALE = 15
    tetromino = Tetromino(constructor_vertices=Shapes.VERTICES_T*SCALE)
    tetromino.center = np.array([0, 0])
    tetromino.print_info()
    tetromino.plot()
    tetromino.rotate(90)
    tetromino.translate(np.array([0, 100]))
    tetromino.print_info()
    tetromino.plot()

if __name__ == '__main__':
    pass