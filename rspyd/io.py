import open3d as o3d
from numpy import loadtxt


def read_cloud(path: str) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud()
    points = loadtxt(path, usecols=(0, 1, 2), delimiter=' ')
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud
