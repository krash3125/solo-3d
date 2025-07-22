import numpy as np
import open3d as o3d
from PIL import Image
import os

# Set your data directory
data_dir = "../anycam_output2/"
output_dir = "../anycam_output3/"

# flip poses
flip_z = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]
)


os.makedirs(output_dir, exist_ok=True)

# Load files
depths = np.load(os.path.join(data_dir, "depths.npy"))  # (N, 1, H, W)
depths = np.squeeze(depths)  # → (N, H, W)
poses = np.load(os.path.join(data_dir, "trajectory.npy"))  # (N, 4, 4)
intrinsics = np.load(os.path.join(data_dir, "projection.npy"))  # (3, 3) or (N, 3, 3)
frame_files = sorted(
    [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".png")]
)

# Expand intrinsics if needed
if intrinsics.ndim == 2:
    intrinsics = np.repeat(intrinsics[None, :, :], len(depths), axis=0)

all_points = []
all_colors = []

for i in range(len(depths)):
    depth = depths[i]
    pose = poses[i] @ flip_z
    K = intrinsics[i]
    color_img = np.array(Image.open(frame_files[i])).astype(np.float32) / 255.0

    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create 2D grid
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x, y = x.flatten(), y.flatten()
    z = depth.flatten()

    # Filter valid depths
    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]

    # Project to 3D
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z
    points_cam = np.vstack((X, Y, Z, np.ones_like(Z)))  # (4, N)

    # Transform to world coordinates
    points_world = (pose @ points_cam)[:3]  # (3, N)

    # Get corresponding RGB
    colors = color_img[y, x]

    all_points.append(points_world.T)
    all_colors.append(colors)

# Merge all frames' data
points = np.concatenate(all_points, axis=0)
colors = np.concatenate(all_colors, axis=0)

# Create point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

print(f"Original point cloud has {len(pcd.points)} points")

# Downsample to reduce noise
# pcd = pcd.voxel_down_sample(voxel_size=0.005)
pcd = pcd.voxel_down_sample(voxel_size=0.007)
# pcd = pcd.voxel_down_sample(voxel_size=0.001)
print(f"Downsampled point cloud has {len(pcd.points)} points")

# Clean up outliers
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# Estimate and orient normals
print("Estimating and orienting normals...")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
)
pcd.orient_normals_consistent_tangent_plane(k=30)

# Poisson reconstruction
print("Running Poisson reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9
)

# Crop mesh to original point cloud bounds
bbox = pcd.get_axis_aligned_bounding_box()
mesh = mesh.crop(bbox)

# Clean and compute normals
mesh.remove_duplicated_vertices()
mesh.remove_degenerate_triangles()
mesh.compute_vertex_normals()

# Transfer color from point cloud using nearest neighbor
print("Transferring colors to mesh vertices...")
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
vertex_colors = []

for vertex in mesh.vertices:
    _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
    vertex_colors.append(pcd.colors[idx[0]])

mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

# Save output
o3d.io.write_point_cloud(os.path.join(output_dir, "colored_point_cloud.ply"), pcd)
o3d.io.write_triangle_mesh(os.path.join(output_dir, "colored_mesh.ply"), mesh)
print("Saved: colored_point_cloud.ply and colored_mesh.ply")

# Visualize
o3d.visualization.draw_geometries([mesh])
