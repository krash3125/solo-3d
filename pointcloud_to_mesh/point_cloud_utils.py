import open3d as o3d


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Failed to load or empty point cloud: {path}")
    print(f"[INFO] Loaded point cloud with {len(pcd.points)} points.")
    return pcd
