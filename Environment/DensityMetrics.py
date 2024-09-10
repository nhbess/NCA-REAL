import numpy as np
from scipy.stats import gaussian_kde

class DensityMetrics:
    @staticmethod
    def sum_inverse_distances_with_radius(chosen_point, circle_centers):
        distances = np.sqrt((chosen_point[0] - circle_centers[:, 0])**2 + (chosen_point[1] - circle_centers[:, 1])**2)
        return 1-np.sum(distances)
    
    @staticmethod
    def nearest_neighbor_density_with_radius(chosen_point, circle_centers):
        distances = np.sqrt((chosen_point[0] - circle_centers[:, 0])**2 + (chosen_point[1] - circle_centers[:, 1])**2)
        return 1 / (1 + np.mean(distances))

    @staticmethod
    def exponential(chosen_point, circle_centers):
        BW=0.3
        
        distances = np.sqrt((chosen_point[0] - circle_centers[:, 0])**2 + (chosen_point[1] - circle_centers[:, 1])**2)
        kde = gaussian_kde(distances, bw_method=BW)
        concentration = kde.evaluate([0])
        return concentration[0]

    @staticmethod
    def kernel_density_estimation_double(chosen_point, circle_centers):
        BW1 = 0.25
        BW2 = 0.75
        
        distances1 = np.sqrt((chosen_point[0] - circle_centers[:, 0])**2 + (chosen_point[1] - circle_centers[:, 1])**2)
        distances2 = np.sqrt((chosen_point[0] - circle_centers[:, 0])**2 + (chosen_point[1] - circle_centers[:, 1])**2)
        
        kde1 = gaussian_kde(distances1, bw_method=BW1)
        kde2 = gaussian_kde(distances2, bw_method=BW2)
        concentration = kde1.evaluate([0])+ kde2.evaluate([0])
        return concentration[0]
    
    @staticmethod
    def closest_distance(chosen_point, circle_centers):
        distances = np.sqrt((chosen_point[0] - circle_centers[:, 0])**2 + (chosen_point[1] - circle_centers[:, 1])**2)
        return 1-(np.sqrt(min(distances)))
    
    @staticmethod
    def exponential(chosen_point, circle_centers):
        BW1=0.5
        BW2=0.1
        
        distances = np.sqrt((chosen_point[0] - circle_centers[:, 0])**2 + (chosen_point[1] - circle_centers[:, 1])**2)
        exp1 = np.exp(-distances**2/(BW1**2))
        exp2 = np.exp(-distances/(BW2))
        
        concentration = np.sum(exp1) + np.sum(exp2)
        return concentration
    