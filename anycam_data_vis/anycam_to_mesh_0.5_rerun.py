import numpy as np
import os
from PIL import Image
import rerun as rr  # Import rerun

# --- Configuration ---
# Set your data directory where AnyCam outputs (depths.npy, trajectory.npy,
# projection.npy, and color .png images) are located.
data_dir = "../anycam_output2/"

# Set the output directory for the saved point cloud and mesh files.
output_dir = "../anycam_output3/"
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# --- Data Loading ---
print(f"Loading data from: {data_dir}")

try:
    # Load depth maps (N, 1, H, W) -> squeeze to (N, H, W)
    depths = np.load(os.path.join(data_dir, "depths.npy"))
    depths = np.squeeze(depths)

    # Load camera poses (N, 4, 4) - assumed to be camera-to-world transformation matrices
    poses = np.load(os.path.join(data_dir, "trajectory.npy"))

    # Load camera intrinsics (3, 3) or (N, 3, 3)
    intrinsics = np.load(os.path.join(data_dir, "projection.npy"))

    # Find all PNG image files in the data directory for color information
    frame_files = sorted(
        [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".png")]
    )

except FileNotFoundError as e:
    print(f"Error: One or more input files not found. Please check paths. {e}")
    print("Attempting to create dummy files for demonstration.")
    # --- Dummy Data Creation (for demonstration if files are missing) ---
    num_frames_dummy = 5
    H_dummy, W_dummy = 189, 189  # Based on your provided shapes

    # Create dummy depths (N, H, W)
    depths = (
        np.random.rand(num_frames_dummy, H_dummy, W_dummy) * 10.0 + 0.1
    )  # Ensure positive depths

    # Create dummy poses (N, 4, 4) with some movement
    poses = np.array([np.eye(4) for _ in range(num_frames_dummy)], dtype=np.float32)
    for i in range(num_frames_dummy):
        poses[i, 0, 3] = i * 0.1  # Translate along X
        poses[i, 1, 3] = np.sin(i * 0.5) * 0.05  # Translate along Y with sine wave
        # Optional: add slight rotation
        theta = i * 0.01
        c, s = np.cos(theta), np.sin(theta)
        rot_z = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        poses[i] = poses[i] @ rot_z

    # Create dummy intrinsics (3, 3)
    intrinsics = np.array(
        [[300.0, 0.0, W_dummy / 2.0], [0.0, 300.0, H_dummy / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    # Create dummy PNG images
    frame_files = []
    dummy_image_dir = os.path.join(data_dir, "dummy_images")
    os.makedirs(dummy_image_dir, exist_ok=True)
    for i in range(num_frames_dummy):
        dummy_img_path = os.path.join(dummy_image_dir, f"frame_{i:04d}.png")
        dummy_img = Image.fromarray(
            (np.random.rand(H_dummy, W_dummy, 3) * 255).astype(np.uint8)
        )
        dummy_img.save(dummy_img_path)
        frame_files.append(dummy_img_path)

    print("Dummy files created. Please replace them with your actual AnyCam outputs.")
    # End dummy data creation

except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()  # Exit if critical data cannot be loaded or created

# Expand intrinsics if it's a single (3, 3) matrix for all frames
if intrinsics.ndim == 2:
    intrinsics = np.repeat(intrinsics[None, :, :], len(depths), axis=0)

# --- Point Cloud Reconstruction ---
all_points = []
all_colors = []

print(f"Processing {len(depths)} frames...")
for i in range(len(depths)):
    depth = depths[i]
    pose = poses[i]
    K = intrinsics[i]

    # Load color image for the current frame
    if i < len(frame_files):
        try:
            color_img = np.array(Image.open(frame_files[i])).astype(np.float32) / 255.0
        except FileNotFoundError:
            print(
                f"Warning: Color image {frame_files[i]} not found. Using dummy color."
            )
            color_img = np.random.rand(
                depth.shape[0], depth.shape[1], 3
            )  # Fallback dummy color
    else:
        print(f"Warning: No color image found for frame {i}. Using dummy color.")
        color_img = np.random.rand(
            depth.shape[0], depth.shape[1], 3
        )  # Fallback dummy color

    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create 2D pixel coordinate grid
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))
    x_coords, y_coords = x_coords.flatten(), y_coords.flatten()
    z_depths = depth.flatten()

    # Filter valid depths (positive values)
    valid_indices = z_depths > 0
    x_valid, y_valid, z_valid = (
        x_coords[valid_indices],
        y_coords[valid_indices],
        z_depths[valid_indices],
    )

    if len(x_valid) == 0:
        print(f"  No valid depth points in frame {i}. Skipping.")
        continue

    # Unproject to 3D points in camera coordinates
    X_cam = (x_valid - cx) * z_valid / fx
    Y_cam = (y_valid - cy) * z_valid / fy
    Z_cam = z_valid
    points_cam = np.vstack((X_cam, Y_cam, Z_cam, np.ones_like(Z_cam)))  # (4, N_valid)

    # Transform points from camera coordinates to world coordinates
    # P_world = T_world_camera @ P_camera
    points_world = (pose @ points_cam)[:3, :].T  # (N_valid, 3)

    # Get corresponding RGB colors for valid points
    colors = color_img[y_valid, x_valid]  # (N_valid, 3)

    all_points.append(points_world)
    all_colors.append(colors)

if not all_points:
    print("No points were reconstructed across all frames. Exiting.")
    exit()

# Merge all frames' data into single arrays
points = np.concatenate(all_points, axis=0)
colors = np.concatenate(all_colors, axis=0)

print(f"Total reconstructed points before processing: {len(points)}")

# --- Point Cloud Processing (using NumPy for operations that don't require Open3D) ---
# Note: For advanced processing like normal estimation or Poisson reconstruction,
# Open3D is still the standard. If you want to visualize *only* the raw
# points and colors in Rerun, you can skip the Open3D steps.
# However, to get a mesh, Open3D is currently necessary.

# We'll use Open3D for the processing steps that lead to a mesh,
# then log the final results to Rerun.
try:
    import open3d as o3d

    print("Open3D found. Proceeding with advanced point cloud processing.")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Original point cloud has {len(pcd.points)} points")

    # Downsample to reduce noise and density
    # Increased voxel_size to make downsampling more aggressive, potentially simplifying for Poisson.
    voxel_size = 0.01  # Changed from 0.007 to 0.01
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(
        f"Downsampled point cloud has {len(pcd.points)} points (voxel_size={voxel_size})"
    )

    # Clean up outliers
    # Adjusted std_ratio to make outlier removal slightly more aggressive.
    nb_neighbors = 20
    std_ratio = 1.5  # Changed from 2.0 to 1.5
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    print(f"Point cloud after outlier removal has {len(pcd.points)} points")

    # Estimate and orient normals
    print("Estimating and orienting normals...")
    # Increased radius and max_nn for normal estimation to improve robustness for sparse data.
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=60
        )  # radius changed from 0.05 to 0.1, max_nn from 30 to 60
    )
    pcd.orient_normals_consistent_tangent_plane(k=30)

    # Poisson reconstruction to create a mesh
    print("Running Poisson reconstruction...")
    # Reduced 'depth' parameter to create a coarser mesh, which is more robust to noise and gaps.
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=6  # Changed from 7 to 6
    )

    # Crop mesh to original point cloud bounds
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # Clean and compute normals for the mesh
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    # Added removal of non-manifold edges for better mesh integrity.
    mesh.remove_non_manifold_edges()  # New cleaning step
    mesh.compute_vertex_normals()

    # Transfer color from point cloud using nearest neighbor
    print("Transferring colors to mesh vertices...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    vertex_colors = []

    for vertex in mesh.vertices:
        _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
        vertex_colors.append(pcd.colors[idx[0]])

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # Save output files (PLY format is good for 3D data)
    o3d.io.write_point_cloud(os.path.join(output_dir, "colored_point_cloud.ply"), pcd)
    o3d.io.write_triangle_mesh(os.path.join(output_dir, "colored_mesh.ply"), mesh)
    print(
        f"Saved: {os.path.join(output_dir, 'colored_point_cloud.ply')} and {os.path.join(output_dir, 'colored_mesh.ply')}"
    )

    # --- Rerun Visualization ---
    # Initialize Rerun. This will launch the Rerun viewer if it's not already running.
    rr.init("AnyCam 4D Reconstruction", spawn=True)

    # Log the processed point cloud
    print("\nLogging processed point cloud to Rerun...")
    rr.log(
        "reconstruction/processed_point_cloud",
        rr.Points3D(positions=np.asarray(pcd.points), colors=np.asarray(pcd.colors)),
    )

    # Log the reconstructed mesh
    print("Logging reconstructed mesh to Rerun...")
    rr.log(
        "reconstruction/mesh",
        rr.Mesh3D(
            vertex_positions=np.asarray(mesh.vertices),
            triangle_indices=np.asarray(mesh.triangles),
            vertex_colors=np.asarray(mesh.vertex_colors),
            vertex_normals=np.asarray(
                mesh.vertex_normals
            ),  # Include normals for better shading
        ),
    )

    # Optionally, log the camera poses as a trajectory
    print("Logging camera trajectory to Rerun...")
    for i, pose_matrix in enumerate(poses):
        # Log the camera's position and orientation for each frame
        # Rerun expects world_from_view, which is the camera-to-world transform
        rr.set_time("frame", i)  # Corrected: Removed 'sequence_timeline='
        rr.log(
            "cameras/camera_trajectory",
            rr.Transform3D(
                matrix=pose_matrix,
                from_parent=True,  # This transform is from the world origin to the camera
            ),
            rr.Pinhole(
                image_from_camera=intrinsics[i] if intrinsics.ndim == 3 else intrinsics,
                width=W,
                height=H,
            ),
            # Add a visual representation of the camera
            rr.ViewCoordinates.RDF,  # Rerun's default camera orientation (Right, Down, Forward)
        )
        # Log a small point at the camera location for visibility
        rr.log(
            f"cameras/camera_trajectory/frame_{i}",
            rr.Points3D(
                positions=pose_matrix[:3, 3].reshape(1, 3),
                radii=0.01,
                colors=[0, 255, 0],
            ),
        )


except ImportError:
    print(
        "Open3D not found. Skipping advanced point cloud processing and mesh generation."
    )
    print("Only raw point cloud will be available if you choose to visualize it.")
    # If Open3D is not available, you can still log the raw points:
    rr.init("AnyCam 4D Reconstruction (Raw Points)", spawn=True)
    rr.log(
        "reconstruction/raw_point_cloud", rr.Points3D(positions=points, colors=colors)
    )
except Exception as e:
    print(f"An error occurred during Open3D/Rerun processing: {e}")
