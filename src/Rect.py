import numpy as np
from scipy.spatial.transform import Rotation
import math


class Rect3d:
    def __init__(self, bottom_left: np.ndarray, top_right: np.ndarray) -> None:
        self.bottom_left = bottom_left
        self.top_right = top_right

    def get_max_size(self) -> int:
        max_size = 0
        for dim in range(3):
            max_size = max(max_size, self.top_right[dim]-self.bottom_left[dim])
        return max_size

    def get_center(self):
        return (self.bottom_left + self.top_right)/2

    def closest_to_point(self, point):
        closest = np.zeros(3)
        for dim in range(3):
            if point[dim] <= self.bottom_left[dim]:
                closest[dim] = self.bottom_left[dim]
            elif point[dim] >= self.top_right[dim]:
                closest[dim] = self.top_right[dim]
            else:
                closest[dim] = point[dim]
        return closest

    def distance_to_point(self, point):
        return np.linalg.norm(point - self.closest_to_point(point))


class Rect:
    def __init__(self, matrix, basis_old, degree: float) -> None:
        basis = np.zeros((3, 3))
        # FIXME i have no idea what this does
        basis[0] = self.rotate(basis_old[0], degree, basis[2])
        basis[1] = self.rotate(basis_old[1], degree, basis[2])
        basis[2] = basis_old[2]
        new_matrix = np.matmul(basis, matrix)
        max = np.amax(new_matrix, axis=1)
        min = np.amin(new_matrix, axis=1)
        w = max[0] - min[0]
        h = max[1] - min[0]
        area = w*h
        self.matrix = matrix
        self.area = area
        self.basis = basis
        self.rect = Rect3d(min, max)

    def rotate(self, v, degree: float, axis):
        rad = math.radians(degree)
        rot = Rotation.from_rotvec(axis*math.radians(degree))
        return rot.apply(v)
