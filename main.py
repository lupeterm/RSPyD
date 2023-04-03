from argparse import ArgumentParser
from rspyd import Detector, ConnectivityGraph, io
import open3d as o3d
import copy
import numpy as np
import time
NUM_NEIGHBORS = 30
EXAMPLE_PATH = 'examples/office.txt'


def main(args):
    benchmarking = args.benchmark
    example = args.example_scene

    path = EXAMPLE_PATH if example else args.cloud_path
    # read unorganized point cloud and estimate normals
    point_cloud = io.read_cloud(path)
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=NUM_NEIGHBORS))

    # reduce point cloud complexity by down sampling
    voxel_size = args.down_sample
    if voxel_size != 0:
        point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    # o3d.visualization.draw_geometries([point_cloud])
    if benchmarking or example:
        t = time.time()

    print('Starting ConnectivityGraph construction!')
    # build connectivity graph
    kd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    connectivity = ConnectivityGraph(len(point_cloud.points))
    for i, point in enumerate(point_cloud.points):
        if len(neighbors := connectivity.get_neighbors(i)) >= NUM_NEIGHBORS:
            connectivity.add_node(i, neighbors)
        else:
            [_, neighbors, _] = kd_tree.search_knn_vector_3d(point, NUM_NEIGHBORS+1)
            connectivity.add_node(i, neighbors[1:])

    if benchmarking or example:
        t1 = time.time()
        pre = t1 - t
        print(f'Building of ConnectivityGraph took {pre} seconds!')
    print('Starting Plane Detection!')

    detector = Detector(point_cloud, connectivity)

    p = detector.detect()

    if benchmarking or example:
        calc = time.time() - t1
        print(f'Plane Detection took {calc} seconds!')

    print(f'Detected {len(p)} planes!')

    pcd_ty = copy.deepcopy(point_cloud).translate((0, 5, 0))
    ps = [pcd_ty]
    colors = [230, 25, 75, 60, 180, 75, 255, 225, 25, 0, 130, 200, 245, 130, 48, 145, 30, 180, 70, 240, 240, 240, 50, 230, 210, 245, 60, 250, 190, 212, 0, 128,
              128, 220, 190, 255, 170, 110, 40, 255, 250, 200, 128, 0, 0, 170, 255, 195, 128, 128, 0, 255, 215, 180, 0, 0, 128, 128, 128, 128, 255, 255, 255]
    colors = np.array(colors).reshape(-1, 3) / 255
    for i, plane in enumerate(list(p)):
        inliers = plane.inliers
        pc = point_cloud.select_by_index(inliers)
        pc.paint_uniform_color(colors[i % len(colors)])
        ps.append(pc)
    o3d.visualization.draw_geometries(ps)


if __name__ == '__main__':
    parser = ArgumentParser('RSPyD')
    parser.add_argument('-c', '--cloud-path', type=str, help='specify path to mesh[.txt | .xyz]. Not needed when --example-scene is set.')
    parser.add_argument('-b', '--benchmark', action='store_true', help='enables benchmarking')
    parser.add_argument('-d', '--down-sample', type=float, default=0.02, help='Defines the voxel size for point cloud size reduction. Disable with 0.')
    parser.add_argument('-e', '--example-scene', action='store_true', help='performs example scene incl. benchmarks and open3d visualization.')
    args = parser.parse_args()
    main(args)
