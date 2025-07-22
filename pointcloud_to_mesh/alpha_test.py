import open3d as o3d
import time


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Failed to load or empty point cloud: {path}")
    print(f"[INFO] Loaded point cloud with {len(pcd.points)} points.")
    return pcd


def alpha_shape_reconstruction(
    pcd: o3d.geometry.PointCloud, alpha: float
) -> o3d.geometry.TriangleMesh:
    print(f"[INFO] Computing alpha shape mesh with alpha = {alpha}")

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


def main(input_path: str, output_path: str, alpha: float):
    pcd = load_point_cloud(input_path)
    mesh = alpha_shape_reconstruction(pcd, alpha)
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

    start_time = time.time()
    main(input, output, alpha)
    end_time = time.time()
    print(f"[INFO] Time taken: {end_time - start_time} seconds")
