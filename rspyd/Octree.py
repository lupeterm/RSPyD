import open3d as o3d
import numpy as np
from typing import List
from .Rect import Rect3d

class Octree:
    def __init__(self, parent=None, center=None, size=None, point_cloud: o3d.geometry.PointCloud = None) -> None:
        if parent != None:
            self.parent = parent
            self.root = self.parent.root
            self.level = self.parent.level + 1
            self.children = [None] * 8
            self.size = size
            self.center = center
        if point_cloud:
            self.cloud = point_cloud
            self.extension: Rect3d = self.calc_extension()
            self.center = self.extension.get_center()
            self.size = self.extension.get_max_size()
            self.indices = np.arange(len(self.cloud.points))
            self.root = self
            self.parent = self
            self.level = 0
            self.leaf_table: List["None | Octree"] = [None] * len(point_cloud.points)
        self.leaf = True
        # get_box_points for corners

    def calc_extension(self) -> Rect3d:
        min_ = np.full(3, 1)*np.inf
        max_ = -min_
        for point in self.cloud.points:
            for dim in range(3):
                min_[dim] = min(min_[dim], point[dim])
                max_[dim] = max(max_[dim], point[dim])
        return Rect3d(min_, max_)

    def get_level(self)->int:
        if self == self.root:
            return 0
        return 1 + self.parent.get_level()


    def partition(self, levels, min_points=1, min_size=0.0) -> None:
        if not self.leaf:
            for child in self.children:
                if child != None:
                    child.partition(levels-1, min_points, min_size)
        else:
            if levels <= 0 or len(self.indices) <= 1 <= min_points or self.size < min_size:
                return
            new_size = self.size*0.5
            new_centers = self.calc_new_centers(new_size)
            self.leaf = False

            # create children
            self.children: List["None| Octree"] = [None]*8

            # split points
            for i in self.indices:

                child_index: int = self.calc_child_index(
                    self.root.cloud.points[i])
                if self.children[child_index] == None:
                    self.children[child_index] = Octree(parent=self, center=new_centers[child_index], size=new_size)
                    self.children[child_index].indices = []
                self.children[child_index].indices.append(i)
                # update current leaf where point is stored
                self.root.leaf_table[i] = self.children[child_index]

            self.indices = []

            for child in self.children:
                if child != None:
                    child.partition(levels-1, min_points, min_size)

    def calc_new_centers(self, new_size):
        new_centers = [0]*8
        for child in range(8):
            d0 = new_size * (((child & 4) >> (2) << 1) - 1)
            d1 = new_size * (((child & 2) >> (1) << 1) - 1)
            d2 = new_size * (((child & 1) << 1) - 1)
            new_centers[child] = self.center + [d0,d1,d2]
        return new_centers

    def calc_child_index(self, point) -> int:
        index = 0
        if point[0] > self.center[0]:
            index += 4
        if point[1] > self.center[1]:
            index += 2
        if point[2] > self.center[2]:
            index +=  1
        return index

    def num_points(self):
        if self.leaf:
            return len(self.indices)
        num_points = 0
        for child in self.children:
            if child != None:
                num_points += child.num_points()
        return num_points

    def get_points(self):
        if self.leaf:
            return self.indices
        points = []
        for child in self.children:
            if child != None:
                points += child.get_points()
        return points

    def get_containing_leaf(self, point):
        return self.root.leaf_table[point]

    def is_root(self) -> bool:
        return self.parent == self