import math
from collections import deque
from typing import List, Literal, Set, Tuple

import numpy as np
import open3d as o3d

from .ConnectivityGraph import ConnectivityGraph
from .Octree import Octree
from .PlanarPatch import PlanarPatch
from .Plane import Plane
from .Rect import Rect, Rect3d
from .StatisticsUtils import StatisticSUtils
from .UnionFind import UnionFind
from .Utils import rspyd_dot, bench, rspyd_orthogonal_base


class IndexedPoint2d:
    def __init__(self, index, point2d):
        self.index = index
        self.point = point2d


class Detector:
    def __init__(self, point_cloud: o3d.geometry.PointCloud, connectivity:  ConnectivityGraph) -> None:
        self.cloud = point_cloud
        self._min_normal_diff = math.cos(math.radians(60.0))
        self._max_distance = math.cos(math.radians(75.0))
        # NOTE semantically:  "max 25% are outliers"
        self._max_outlier_ratio = 0.75
        self.connectivity = connectivity
        self.extension = self.calc_extension()

    def config(self, min_normal_diff, max_distance, min_inlier_ratio):
        self._min_normal_diff = min_normal_diff
        self._max_distance = max_distance
        self._max_outlier_ratio = min_inlier_ratio

    @bench
    def detect(self) -> Set[Plane]:
        planes: Set[Plane] = set()

        self.clear_removed_points()

        min_num_points = int(max(10, len(self.cloud.points)*0.001))
        stat_utils = StatisticSUtils(len(self.cloud.points))

        # start = time()
        patches = []
        octree = Octree(point_cloud=self.cloud)

        self.detect_planar_patches(octree, stat_utils, min_num_points, patches)
        
        self._patch_points: List['None | PlanarPatch'] = [
            None]*len(self.cloud.points)

        for patch in patches:
            for point in patch.points:
                self._patch_points[point] = patch

        changed = True
        while changed:
            patches = self.grow_patches(patches)
            patches = self.merge_patches(patches)
            changed = self.update_patches(patches)
            if not changed:
                break
        patches = self.grow_patches(patches, relaxed=True)

        true_positive_patches: List[PlanarPatch] = []
        fps = 0
        for patch in patches:
            self.delimit_plane(patch)
            if not self.is_false_positive(patch):
                true_positive_patches.append(patch)
            else:
                fps += 1

        patches = true_positive_patches

        for patch in patches:
            plane = patch.plane
            plane.inliers = patch.points
            planes.add(plane)
        return planes

    def detect_planar_patches(self, node: Octree,
                              stat_utils: StatisticSUtils,
                              min_num_points: int,
                              patches: List[PlanarPatch]) -> bool:
        """TODO Diese Funktion könnte ggf über o3d.geometry.octree.traverse() gelöst werden."""
        if node.num_points() < min_num_points:
            return False
        has_planar_patch = False
        node.partition(levels=1, min_points=min_num_points,
                       min_size=0.0)  # equiv as far as i am concerned
        # for child in node.children:
        for i in range(8):
            child = node.children[i]
            if child == None:
                continue
            # print("{}level{}: child-{} has {} samples".format("\t"*child.get_level(), child.get_level(),i, len(child.indices)))
            if self.detect_planar_patches(child, stat_utils, min_num_points, patches):
                has_planar_patch = True
        if not has_planar_patch and node.level > 2:
            patch = PlanarPatch(self.cloud, stat_utils, node.get_points(),
                                self._min_normal_diff, self._max_distance, self._max_outlier_ratio)
            if patch.is_planar():
                patches.append(patch)
                has_planar_patch = True
        return has_planar_patch

    def clear_removed_points(self):
        if not self.cloud:
            return
        self._removed_points = 0
        self._removed = [False]*len(self.cloud.points)
        self._available_points = np.arange(len(self.cloud.points))

    def detect_planar_patches2(self, octree_points, stat_utils: StatisticSUtils,
                               min_num_points: int,
                               patches: List[PlanarPatch],
                               start_level: int = 0) -> bool:
        """Accumulation of planar patches within the point cloud.
        Relies heavily on o3d.geometry.Octree.traverse().

        Note that traverse can only be called from an Octree, but not any other Internal or Leaf node.
        Furthermore, creating new octrees and passing them to this method resets the included point indices.
        For that reason, we pass the list of indices to keep the global structure.
        We pass the max_level and start_level to keep track of the current depth of recursion.
        """
        has_planar_patch = False
        # a node can only be considered planar if #points exceeds a threshold
        if len(octree_points) < min_num_points:
            has_planar_patch = False
            return has_planar_patch

        # create octree for current sample of points
        sub_cloud = self.cloud.select_by_index(octree_points)
        # we only want to traverse one level per call
        octree = o3d.geometry.Octree(max_depth=1)
        octree.convert_from_point_cloud(sub_cloud)

        current_child_indices = []

        def get_child_indices_traversal(node, node_info) -> Literal[True]:
            """
            Traversal function to gather child node indices for given octree node.

            These indices point to the indices within the parent node, so we transform them back to global indices:

            Assume the parent node has the indices [0,1,5,7,12,16,53].

            If a child node has the indices [2,3,5], the values would be [5,7,16] in the global context (as this propagates back to the root element).

            """
            nonlocal current_child_indices
            early_stop = True
            if isinstance(node, o3d.geometry.OctreeInternalPointNode):
                for i in range(8):
                    child = node.children[i]
                    if child is not None:
                        child_global_indices = [node.indices[i]
                                                for i in child.indices]
                        current_child_indices.append(child_global_indices)
                    else:
                        current_child_indices.append([])
            return early_stop

        #  iterate over all child nodes of octree node
        octree.traverse(get_child_indices_traversal)
        # for child_indices in current_child_indices:
        for i in range(8):
            child_indices = current_child_indices[i]
            if child_indices == []:
                continue
            # print("{}level{}: child-{} has {} samples".format("\t"*start_level, start_level,i, len(child_indices)))
            # recursively traverse the octree
            has_planar_child = self.detect_planar_patches2(
                child_indices, stat_utils, min_num_points, patches, start_level+1)
            if has_planar_child:
                has_planar_patch = True
        # if no planar patch has been found yet, and the current node has a depth of at least 3:
        # create and add a patch to the list of patches if it is planar
        if not has_planar_patch and start_level > 2:
            patch = PlanarPatch(self.cloud, stat_utils, octree_points,
                                self._min_normal_diff, self._max_distance, self._max_outlier_ratio)
            if patch.is_planar():
                patches.append(patch)
                has_planar_patch = True
        return has_planar_patch

    def grow_patches(self, patches: List[PlanarPatch], relaxed=False) -> List[PlanarPatch]:
        patches.sort(key=lambda p: p.min_normal_diff(), reverse=True)
        queue = deque()
        for patch in patches:
            if patch.stable:
                continue
            for point in patch.points:
                queue.append(point)
            while len(queue) > 0:
                point = queue.popleft()

                for neighbor in self.connectivity.get_neighbors(point)[:-1]:
                    if self._removed[neighbor] or self._patch_points[neighbor] != None or (not relaxed and patch.is_visited(neighbor)):
                        continue
                    if (not relaxed and patch.is_inlier(neighbor)) or (relaxed and (abs(patch.plane.get_signed_dist_from_surface(self.cloud.points[neighbor])) < patch._max_dist_plane)):
                        queue.append(neighbor)
                        patch.add_point(neighbor)
                        self._patch_points[neighbor] = patch
                    else:
                        patch.visit(neighbor)
        return patches

    def merge_patches(self, patches: List[PlanarPatch]) -> List[PlanarPatch]:
        n = len(patches)
        for i in range(n):
            patches[i].set_index(i)
        graph = np.full(n*n, False)
        disconnected_patches = np.full(n*n, False)
        for i, p in enumerate(patches):
            for j in range(i+1, n):
                # check patch normal deviation
                normal_th = min(p._min_normal_diff,
                                patches[j]._min_normal_diff)
                pp = p.plane
                pj = patches[j].plane
                d = abs(pp.normal[0] * pj.normal[0] + pp.normal[1] *
                        pj.normal[1] + pp.normal[2] * pj.normal[2]) < normal_th
                disconnected_patches[i*n+j] = disconnected_patches[j*n+i] = d

        for patch in patches:
            for point in patch.points:
                neighbors = self.connectivity.get_neighbors(point)
                for neighbor in neighbors:
                    neighbor_patch = self._patch_points[neighbor]
                    if (patch == neighbor_patch
                        or neighbor_patch == None
                        or graph[neighbor_patch.index * n + patch.index]
                        or graph[patch.index * n+neighbor_patch.index]
                        # if disconnected
                        or disconnected_patches[patch.index*n + neighbor_patch.index]
                        or patch.is_visited(neighbor)
                            or neighbor_patch.is_visited(point)):
                        continue
                    patch.visit(neighbor)
                    neighbor_patch.visit(point)
                    p1 = self.cloud.points[point]
                    pp = patch.plane
                    pn = neighbor_patch.plane
                    p1_n = self.cloud.normals[point]
                    p2 = self.cloud.points[neighbor]
                    p2_n = self.cloud.normals[neighbor]
                    distance_th = max(patch._max_dist_plane,
                                      neighbor_patch._max_dist_plane)
                    normal_th = min(patch._min_normal_diff,
                                    neighbor_patch._min_normal_diff)
                    graph[patch.index * n+neighbor_patch.index] = (
                        abs(pp.normal[0] * p2_n[0]+pp.normal[1] * p2_n[1]+pp.normal[2] * p2_n[2]) > normal_th
                        and abs(pn.normal[0] * p1_n[0]+pn.normal[1] * p1_n[1]+pn.normal[2] * p1_n[2]) > normal_th
                        and abs(pp.get_signed_dist_from_surface(p2)) < distance_th
                        and abs(pn.get_signed_dist_from_surface(p1)) < distance_th)
        ufind = UnionFind(n)
        # iterate through graph and join nodes
        for i in range(n):
            for j in range(n):
                if graph[i*n+j] or graph[j*n+i]:  # if passed merge criteria
                    ufind.join(i, j)

        largest_patch = np.arange(n)
        for i in range(n):
            root = ufind.root(i)
            if len(patches[largest_patch[root]].points) < len(patches[i].points):
                largest_patch[root] = i

        for i in range(n):
            root = largest_patch[ufind.root(i)]
            if root != i:
                for point in patches[i].points:
                    patches[root].add_point(point)
                    self._patch_points[point] = patches[root]
                patches[root]._max_dist_plane = max(
                    patches[root]._max_dist_plane, patches[i]._max_dist_plane)
                patches[root]._min_normal_diff = min(
                    patches[root]._min_normal_diff, patches[i]._min_normal_diff)
                patches[i] = None
        return list(filter(None, patches))

    def update_patches(self, patches: List[PlanarPatch]) -> bool:
        changed = False
        for patch in patches:
            if patch._num_new_points > (len(patch.points) - patch._num_new_points)/2:
                patch.update_plane()
                patch.stable = False
                changed = True
            else:
                patch.stable = True
        return changed

    def delimit_plane(self, patch: PlanarPatch) -> None:
        outlier = self.get_plane_outlier(patch)
        n = patch.plane.normal
        base_u,base_v = rspyd_orthogonal_base(n)
        basis = np.zeros((3, 3))
        for dim in range(3):
            basis[0, dim] = base_u[dim]
            basis[1, dim] = base_v[dim]
            basis[2, dim] = n[dim]

        # FIXME breaks if no outliers are found
        # i do not expect an outlier free mesh, but this should still be handled
        matrix = np.zeros((3, len(outlier)))
        for i, outl in enumerate(outlier):
            matrix[:, i] = self.cloud.points[outl][:]

        min_angle = 0
        max_angle = 90
        while max_angle - min_angle > 5:
            mid = (max_angle+min_angle) / 2
            left = (min_angle+mid) / 2
            right = (max_angle+mid) / 2
            if len(outlier) == 0:
                max_angle = min_angle = mid
                matrix = np.zeros((3, 1))
                matrix[:, 0] = np.array([9999999, 9999999, 9999999])[:]
                break
            left_rect = Rect(matrix, basis, left)
            right_rect = Rect(matrix, basis, right)
            if left_rect.area < right_rect.area:
                max_angle = mid
            else:
                min_angle = mid
        patch.rect = Rect(matrix, basis, (min_angle+max_angle)/2)
        center = patch.plane.center
        min_basis_u = patch.rect.basis[0]
        min_basis_v = patch.rect.basis[1]
        center -= min_basis_u * rspyd_dot(min_basis_u, center)
        center -= min_basis_v * rspyd_dot(min_basis_v, center)
        center += min_basis_u * (patch.rect.rect.bottom_left[0] + patch.rect.rect.top_right[0]) / 2
        center += min_basis_v * (patch.rect.rect.bottom_left[1] + patch.rect.rect.top_right[1]) / 2
        length_u = (patch.rect.rect.top_right[0] - patch.rect.rect.bottom_left[0] / 2)
        length_v = (patch.rect.rect.top_right[1] - patch.rect.rect.bottom_left[1] / 2)
        new_plane = Plane(center, patch.plane.normal,min_basis_u * length_u, min_basis_v*length_v)
        patch.plane = new_plane

    def is_false_positive(self, patch: PlanarPatch) -> bool:
        a = patch.num_updates == 0
        # ignore very small patches
        b = (patch.get_size() / self.extension.get_max_size()) < 0.01
        return a or b

    def is_removed(self, point: int):
        return self._removed[point]

    def get_plane_outlier(self, patch: PlanarPatch) -> List[int]:        
        n = patch.plane.normal
        base_u, base_v = rspyd_orthogonal_base(n)
        projected_points = np.zeros((len(patch.points),2))
        for i, point in enumerate(patch.points):
            pos = self.cloud.points[point]
            projected_points[i,:] = self.project_onto_orthogonal_basis(pos,base_u, base_v)
            # projected_points.append(
                # self.project_onto_orthogonal_basis(pos, base_u, base_v))
        outliers = self.convex_hull(projected_points)
        for i, out in enumerate(outliers):
            outliers[i] = patch.points[out]
        return outliers

    def project_onto_orthogonal_basis(self, vector, b_u, b_v):
        alpha = rspyd_dot(vector, b_u)
        beta = rspyd_dot(vector, b_v)
        return alpha,beta

    def convex_hull(self, points) -> List[int]:
        indices = []
        if len(points) < 3:
            for p in points:
                indices.append(p)

        indexed_points: List[IndexedPoint2d] = []
        for i in range(len(indexed_points)):
            indexed_points.append(IndexedPoint2d(i, points[i]))
        indexed_points.sort(key=lambda x: x.point)

        H: List[IndexedPoint2d] = []
        k = 0
        # build lower hull
        for i in range(len(indexed_points)):
            while k >= 2 and self.convex_hull_cross(H[k-2], H[k-1], indexed_points[i]) <= 0:
                k -= 1
            H[k] = indexed_points[i]
            k += 1

        # build upper hull
        i = len(indexed_points)-1
        t = k+1
        while i > 0:
            while k >= t and self.convex_hull_cross(H[k-2], H[k-1], indexed_points[i-1]) <= 0:
                k -= 1
            H[k] = indexed_points[i-1]
            k += 1
            i -= 1
        H = H[:k]
        indices = [0] * len(H)
        for i, point2d in enumerate(H):
            indices[i] = point2d.index
        return indices

    def convex_hull_cross(self, o: IndexedPoint2d, a: IndexedPoint2d, b: IndexedPoint2d):
        return (a.point[0] - o.point[0]) * (b.point[1] - o.point[1]) - (a.point[1] - o.point[1]) * (b.point[0] - o.point[0])

    def calc_extension(self) -> Rect3d:
        min_ = np.full(3, 1)*np.inf
        max_ = -min_
        for point in self.cloud.points:
            for dim in range(3):
                min_[dim] = min(min_[dim], point[dim])
                max_[dim] = max(max_[dim], point[dim])
        return Rect3d(min_, max_)
