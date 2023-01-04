from numpy import loadtxt
from argparse import ArgumentParser
from rspyd import Detector, ConGraph
import open3d as o3d
import copy
import numpy as np
NUM_NEIGHBORS = 30

if __name__ == '__main__':
    parser = ArgumentParser('RSPyD')
    parser.add_argument('-m', '--mesh-path', type=str,
                        help='specify path to mesh[.txt | .xyz]')
    args = parser.parse_args()

    points = loadtxt(args.mesh_path, usecols=(0, 1, 2), delimiter=' ')

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud = point_cloud.voxel_down_sample(voxel_size=0.08)
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    o3d.visualization.draw_geometries([point_cloud])



    kd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    connectivity = ConGraph(len(point_cloud.points))
    for i, point in enumerate(point_cloud.points):
        if len(neighbors := connectivity.get_neighbors(i)) >= NUM_NEIGHBORS:
            connectivity.add_node(i, neighbors)
        else:
            [_, neighbors, _] = kd_tree.search_knn_vector_3d(
                point_cloud.points[i], NUM_NEIGHBORS+1)
            connectivity.add_node(i, neighbors[1:])
            
    detector = Detector(point_cloud, connectivity)
    detector.config(0.5, 0.258819, 0.75)

    p = detector.detect()
    print(f'Detected {len(p)} planes!')

    pcd_ty = copy.deepcopy(point_cloud).translate((0, 20, 0))
    ps = [pcd_ty]
    ps = []
    colors = [230, 25, 75, 60, 180, 75, 255, 225, 25, 0, 130, 200, 245, 130, 48, 145, 30, 180, 70, 240, 240, 240, 50, 230, 210, 245, 60, 250, 190, 212, 0, 128,
              128, 220, 190, 255, 170, 110, 40, 255, 250, 200, 128, 0, 0, 170, 255, 195, 128, 128, 0, 255, 215, 180, 0, 0, 128, 128, 128, 128, 255, 255, 255, 0, 0, 0]
    colors = np.array(colors).reshape(-1, 3) / 255
    for i, plane in enumerate(list(p)):
        inliers = plane.inliers
        pc = point_cloud.select_by_index(inliers)
        pc.paint_uniform_color(colors[i%len(colors)])
        ps.append(pc)
    o3d.visualization.draw_geometries(ps)