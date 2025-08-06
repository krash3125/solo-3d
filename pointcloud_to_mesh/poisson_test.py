import open3d as o3d
import numpy as np
import time
from .point_cloud_utils import load_point_cloud


def estimate_normals(
    pcd: o3d.geometry.PointCloud, radius: float = 0.03, max_nn: int = 50
):
    print("[INFO] Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    print("[INFO] Orienting normals consistently...")
    pcd.orient_normals_consistent_tangent_plane(k=200)


def poisson_reconstruction(
    pcd: o3d.geometry.PointCloud, depth: int = 8, downsample_voxel_size: float = None
) -> o3d.geometry.TriangleMesh:
    print(f"[INFO] Running Poisson Reconstruction with depth = {depth}")

    # Downsample if voxel_size is provided
    if downsample_voxel_size is not None:
        print(
            f"[INFO] Downsampling point cloud with voxel size = {downsample_voxel_size}"
        )
        pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
        print(f"[INFO] Downsampled point cloud has {len(pcd.points)} points")

    # Estimate normals after downsampling
    estimate_normals(pcd)

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


def main(
    input_path: str, output_path: str, depth: int, downsample_voxel_size: float = None
):
    pcd = load_point_cloud(input_path)

    try:
        mesh = poisson_reconstruction(pcd, depth, downsample_voxel_size)
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
    # Add downsampling voxel size (e.g., 0.01 for aggressive downsampling, 0.001 for light downsampling)
    downsample_voxel_size = None  # Set to a value like 0.01 to enable downsampling

    start_time = time.time()
    main(input, output, depth, downsample_voxel_size)
    end_time = time.time()
    print(f"[INFO] Time taken: {end_time - start_time:.2f} seconds")
