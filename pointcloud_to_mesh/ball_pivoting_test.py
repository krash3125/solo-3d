import open3d as o3d
import time


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Failed to load or empty point cloud: {path}")
    print(f"[INFO] Loaded point cloud with {len(pcd.points)} points.")
    return pcd


def estimate_normals(
    pcd: o3d.geometry.PointCloud, radius: float = 0.01, max_nn: int = 30
):
    print("[INFO] Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_consistent_tangent_plane(100)


def ball_pivoting_reconstruction(
    pcd: o3d.geometry.PointCloud, radii: list, downsample_voxel_size: float = None
) -> o3d.geometry.TriangleMesh:
    print(f"[INFO] Running Ball Pivoting with radii: {radii}")

    # Downsample if voxel_size is provided
    if downsample_voxel_size is not None:
        print(
            f"[INFO] Downsampling point cloud with voxel size = {downsample_voxel_size}"
        )
        pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
        print(f"[INFO] Downsampled point cloud has {len(pcd.points)} points")

    # Estimate normals after downsampling
    estimate_normals(pcd)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    mesh.compute_vertex_normals()
    return mesh


def visualize_geometries(geometries, title="Open3D Viewer"):
    o3d.visualization.draw_geometries(
        geometries, window_name=title, mesh_show_back_face=True
    )


def main(
    input_path: str, output_path: str, radii: list, downsample_voxel_size: float = None
):
    pcd = load_point_cloud(input_path)
    mesh = ball_pivoting_reconstruction(pcd, radii, downsample_voxel_size)
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"[INFO] Mesh saved to: {output_path}")
    # visualize_geometries(
    #     [mesh.translate((1.0, 0, 0))],
    #     title="Ball Pivoting Mesh",
    # )


if __name__ == "__main__":
    input = "./fused_filtered_frames.ply"
    output = "./ball_pivoting_mesh.ply"
    radii = [0.005, 0.01, 0.02]  # Try adjusting based on your point cloud scale
    # Add downsampling voxel size (e.g., 0.01 for aggressive downsampling, 0.001 for light downsampling)
    downsample_voxel_size = None  # Set to a value like 0.01 to enable downsampling

    start_time = time.time()
    main(input, output, radii, downsample_voxel_size)
    end_time = time.time()
    print(f"[INFO] Time taken: {end_time - start_time:.2f} seconds")
