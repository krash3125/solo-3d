import open3d as o3d
import time
from .point_cloud_utils import load_point_cloud


def alpha_shape_reconstruction(
    pcd: o3d.geometry.PointCloud, alpha: float, downsample_voxel_size: float = None
) -> o3d.geometry.TriangleMesh:
    print(f"[INFO] Computing alpha shape mesh with alpha = {alpha}")

    # Downsample if voxel_size is provided
    if downsample_voxel_size is not None:
        print(
            f"[INFO] Downsampling point cloud with voxel size = {downsample_voxel_size}"
        )
        pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
        print(f"[INFO] Downsampled point cloud has {len(pcd.points)} points")

    # Create tetrahedral mesh and point mapping (can be reused)
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)

    # ✅ Pass both tetra_mesh and pt_map
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra_mesh, pt_map
    )
    mesh.compute_vertex_normals()
    return mesh


def visualize_geometries(geometries, title="Open3D Viewer"):
    o3d.visualization.draw_geometries(
        geometries, window_name=title, mesh_show_back_face=True
    )


def main(
    input_path: str, output_path: str, alpha: float, downsample_voxel_size: float = None
):
    pcd = load_point_cloud(input_path)
    mesh = alpha_shape_reconstruction(pcd, alpha, downsample_voxel_size)
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"[INFO] Mesh saved to: {output_path}")
    # visualize_geometries(
    #     # [pcd.translate((0, 0, 0)), mesh.translate((1.0, 0, 0))],
    #     [mesh.translate((1.0, 0, 0))],
    #     title="Point Cloud & Alpha Shape Mesh",
    # )


if __name__ == "__main__":
    input = "./fused_filtered_frames.ply"
    output = "./alpha_mesh.ply"
    alpha = 0.03
    # Add downsampling voxel size (e.g., 0.01 for aggressive downsampling, 0.001 for light downsampling)
    downsample_voxel_size = None  # Set to a value like 0.01 to enable downsampling

    start_time = time.time()
    main(input, output, alpha, downsample_voxel_size)
    end_time = time.time()
    print(f"[INFO] Time taken: {end_time - start_time} seconds")
