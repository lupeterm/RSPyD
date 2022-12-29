import numpy as np
from typing import List
import open3d as o3d


class Plane:
    def __init__(self, center=None, normal=None, basis_u=np.zeros(3), basis_v=np.zeros(3)) -> None:
        if isinstance(center, np.ndarray):
            self.center = center
        else:
            self.center = np.zeros(3)
        if isinstance(normal, np.ndarray):
            self.normal = normal
        else:
            self.normal = np.ones(3)
        self.inliers: List[int] = []
        self._distance_origin = -np.dot(self.normal, self.center)
        self.basis_u = basis_u
        self.basis_v = basis_v
        self.inlier = []

    def visualize(self):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.array(self.inliers))
        pc.paint_uniform_color((1, 0, 0))
        o3d.visualization.draw_geometries([pc])

    def get_signed_dist_from_surface(self, point):
        return (self.normal[0]*point[0]+self.normal[1]*point[1]+self.normal[2]*point[2]) + self._distance_origin

    def get_rotation(self):
        """
        Calculates the matrix that rotates the plane's normal vector to match the unit vector facing straight up. 

        based on:
        https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
        """
        origin = np.array([0, 1, 0])
        norm_normal = self.normal / np.linalg.norm(self.normal)
        v = np.cross(origin, norm_normal)
        c = np.dot(origin, norm_normal)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + \
            kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
