import open3d as o3d
import numpy as np
import time


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Failed to load or empty point cloud: {path}")
    print(f"[INFO] Loaded point cloud with {len(pcd.points)} points.")
    return pcd


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, downsample_voxel_size=0.003):
    print(f"[INFO] Downsampling point cloud with voxel size = {downsample_voxel_size}")
    pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)

    print("[INFO] Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50)
    )
    print("[INFO] Orienting normals consistently...")
    pcd.orient_normals_consistent_tangent_plane(k=200)

    return pcd


def poisson_reconstruction(
    pcd: o3d.geometry.PointCloud, depth: int = 8
) -> o3d.geometry.TriangleMesh:
    print(f"[INFO] Running Poisson Reconstruction with depth = {depth}")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    mesh.compute_vertex_normals()

    print("[INFO] Filtering low-density vertices...")
    densities_np = np.asarray(densities)
    threshold = np.percentile(densities_np, 5)
    vertices_to_keep = densities_np > threshold
    mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])

    return mesh


def visualize_geometries(geometries, title="Open3D Viewer"):
    o3d.visualization.draw_geometries(
        geometries, window_name=title, mesh_show_back_face=True
    )


def main(input_path: str, output_path: str, depth: int):
    pcd = load_point_cloud(input_path)
    pcd = preprocess_point_cloud(pcd)

    try:
        mesh = poisson_reconstruction(pcd, depth)
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"[INFO] Mesh saved to: {output_path}")
        # visualize_geometries(
        #     [mesh.translate((1.0, 0, 0))],
        #     title="Poisson Mesh",
        # )
    except Exception as e:
        print(f"[ERROR] Poisson reconstruction failed: {e}")


if __name__ == "__main__":
    input = "./fused_filtered_frames.ply"
    output = "./poisson_mesh.ply"
    depth = 8  # Safer value than 9

    start_time = time.time()
    main(input, output, depth)
    end_time = time.time()
    print(f"[INFO] Time taken: {end_time - start_time:.2f} seconds")
