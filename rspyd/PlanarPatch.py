from typing import List, Dict
from .StatisticsUtils import StatisticSUtils
import open3d as o3d
import numpy as np
from .Plane import Plane
from .Utils import rspyd_dot, rspyd_norm, rspyd_orthogonal_base
from .Rect import Rect3d, Rect
import copy

class PlanarPatch:
    def __init__(self, point_cloud: o3d.geometry.PointCloud,
                 stat_utils: StatisticSUtils,
                 points: List[int],
                 min_allowed_normal: float,
                 max_allowed_distance: float,
                 outlier_ratio: float) -> None:
        self._point_cloud = point_cloud
        self.points = points
        self.normals = np.take(np.asarray(
            self._point_cloud.normals), self.points, axis=0)
        self.pts = np.take(np.asarray(
            self._point_cloud.points), self.points, axis=0)
        self._stat_utils = stat_utils
        self._original_size = self.get_size()
        self.plane = Plane()
        self._outlier_ratio = outlier_ratio
        self._outliers: Dict[int, bool] = {}
        self._num_new_points = 0
        self._visited: List[bool] = []
        self._visited_set = set()
        self._min_allowed_normal = min_allowed_normal
        self._max_allowed_distance = max_allowed_distance
        self.stable = False
        self.num_updates = 0
        self.rect: "None | Rect" = None
        self.used_visited2 = False

    def visualize(self):
        pc = self.get_cloud_for_vis()
        pcd_ty = copy.deepcopy(self._point_cloud).translate((0, 10, 0))
        o3d.visualization.draw_geometries([pc,pcd_ty])        

    def get_cloud_for_vis(self):
        pc = self._point_cloud.select_by_index(self.points)
        pc.paint_uniform_color(np.random.rand(3))
        return pc

    def get_size(self) -> int:
        rect = self.get_rect()
        return rect.get_max_size()

    def get_rect(self) -> Rect3d:
        mins = np.min(self.pts, axis=0)
        maxs = np.max(self.pts, axis=0)
        return Rect3d(mins, maxs)

    def get_plane(self) -> Plane:
        center = np.partition(self.pts, axis=0, kth=len(self.pts)//2)[len(self.pts)//2]
        normal = np.partition(self.normals, axis=0, kth=len(self.normals)//2)[len(self.normals)//2]
        normal = rspyd_norm(normal)
        return Plane(center, normal)

    def is_planar(self) -> bool:
        self.plane = self.get_plane()
        self._min_normal_diff = self.get_min_normal_diff()
        if not self.is_normal_valid():
            return False
        num_outliers = 0
        for i, p in enumerate(self.points):
            is_outlier = self._stat_utils.databuffer[i] < self._min_normal_diff
            self._outliers[p] = is_outlier
            if is_outlier:
                num_outliers += 1
        if num_outliers > len(self.points)*self._outlier_ratio:
            return False

        self._max_dist_plane = self.get_max_plane_dist()
        if not self.is_dist_valid():
            return False
        num_outliers = 0
        for i, p in enumerate(self.points):
            is_outlier = self._outliers[p] or self._stat_utils.databuffer[i] > self._max_dist_plane
            self._outliers[i] = is_outlier
            if is_outlier:
                num_outliers += 1
        if num_outliers < len(self.points)*self._outlier_ratio:
            self.remove_outliers()
            return True
        return False

    def get_min_normal_diff(self):
        self._stat_utils.size(len(self.points))
        pn = self.plane.normal
        for i, normal in enumerate(self.normals):
            self._stat_utils.databuffer[i] = abs(normal[0]*pn[0] + normal[1]*pn[1] + normal[2]*pn[2])
        min, _ = self._stat_utils.get_min_max_R(3)
        return min

    def is_normal_valid(self):
        return self._min_normal_diff > self._min_allowed_normal

    def get_max_plane_dist(self):
        self._stat_utils.size(len(self.points))
        for i, point in enumerate(self.pts):
            self._stat_utils.databuffer[i] = abs(self.plane.get_signed_dist_from_surface(point))
        _, max = self._stat_utils.get_min_max_R(3)
        return max

    def is_dist_valid(self) -> bool:
        n = self.plane.normal
        base_u, base_v = rspyd_orthogonal_base(n)
        extreme = base_u * self._original_size + n * self._max_dist_plane
        extreme_n = rspyd_norm(extreme)
        a = abs(rspyd_dot(n,extreme_n))
        return a < self._max_allowed_distance

    def remove_outliers(self):
        self.points = list(
            filter(lambda p: not self._outliers[p], self.points))
        self._outliers = dict()

    def min_normal_diff(self):
        return self._min_normal_diff

    def max_plane_dist(self):
        return self._max_dist_plane

    def is_stable(self) -> bool:
        return self.stable

    def is_inlier(self, point: int):
        n = self.plane.normal
        p = self._point_cloud.normals[point]
        d = abs(rspyd_dot(n,p))
        return d > self._min_normal_diff and self._max_dist_plane > d + self.plane._distance_origin

    def visit(self, point: int) -> None:
        if len(self._visited) > 0:
            self._visited[point] = True
        else:
            self._visited_set.add(point)

    def is_visited(self, point) -> bool:
        if len(self._visited) > 0:
            return self._visited[point]
        return point in self._visited_set

    def add_point(self, point: int) -> None:
        self.points.append(point)
        self._num_new_points += 1
        self._outliers[point] = False

    def set_index(self, index: int) -> None:
        self.index = index

    def update_plane(self):
        self.plane = self.get_plane()
        self._visited_set = set()
        if self.num_updates > 1:
            self.used_visited2 = True
            if len(self._visited) == 0:
                self._visited = [False]*len(self._point_cloud.points)
            else:
                self._visited = [False]*len(self._visited)
        self._num_new_points = 0
        self.num_updates += 1
