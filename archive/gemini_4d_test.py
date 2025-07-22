import numpy as np
import os
import open3d as o3d


def unproject_pixel_to_3d(u, v, depth, fx, fy, cx, cy):
    """
    Unprojects a 2D pixel coordinate with its depth into a 3D point
    in the camera's coordinate system using full intrinsic parameters.

    Args:
        u (float): X-coordinate of the pixel.
        v (float): Y-coordinate of the pixel.
        depth (float): Depth value at the pixel.
        fx (float): Focal length in x-direction.
        fy (float): Focal length in y-direction.
        cx (float): Principal point X-coordinate.
        cy (float): Principal point Y-coordinate.

    Returns:
        numpy.ndarray: A 3D point (X, Y, Z) in the camera's coordinate system.
    """
    # Z-coordinate is the depth itself
    Z_c = depth
    # X and Y coordinates are derived using the pinhole camera model
    X_c = (u - cx) * Z_c / fx
    Y_c = (v - cy) * Z_c / fy
    return np.array([X_c, Y_c, Z_c])


def perform_4d_reconstruction(
    depths_path: str,
    trajectory_path: str,
    uncertainties_path: str,
    projection_path: str,  # Now expects a 3x3 intrinsic matrix
    output_filename: str = "reconstructed_point_cloud.npy",
    uncertainty_threshold: float = 0.05,
):
    """
    Performs 4D reconstruction by combining depth maps, camera poses,
    and uncertainty maps from AnyCam outputs.

    Args:
        depths_path (str): Path to the depths.npy file (shape: [num_frames, 1, H, W]).
        trajectory_path (str): Path to the trajectory.npy file (shape: [num_frames, 4, 4]).
                                Each 4x4 matrix is assumed to be an absolute camera-to-world pose.
        uncertainties_path (str): Path to the uncertainties.npy file (shape: [num_frames, 1, H, W]).
        projection_path (str): Path to the projection.npy file. Assumed to contain
                                a 3x3 camera intrinsic matrix.
        output_filename (str): Name for the output .npy file containing the aggregated 3D points.
        uncertainty_threshold (float): Maximum uncertainty value for a point to be included
                                       in the reconstruction (points with higher uncertainty
                                       are considered dynamic and filtered out).
    """
    print(
        f"Loading data from: {depths_path}, {trajectory_path}, {uncertainties_path}, {projection_path}"
    )

    try:
        depths = np.load(depths_path)
        trajectory = np.load(trajectory_path)
        uncertainties = np.load(uncertainties_path)
        intrinsic_matrix = np.load(projection_path)
    except FileNotFoundError as e:
        print(f"Error: One or more input files not found. Please check paths. {e}")
        return
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return

    # Validate shapes and types
    if depths.ndim != 4 or depths.shape[1] != 1:
        print(
            f"Error: depths.npy expected 4 dimensions (frames, 1, H, W), got {depths.ndim} or incorrect second dim."
        )
        return
    if trajectory.ndim != 3 or trajectory.shape[1:] != (4, 4):
        print(
            f"Error: trajectory.npy expected 3 dimensions (frames, 4, 4), got {trajectory.ndim} or incorrect inner shape."
        )
        return
    if uncertainties.ndim != 4 or uncertainties.shape[1] != 1:
        print(
            f"Error: uncertainties.npy expected 4 dimensions (frames, 1, H, W), got {uncertainties.ndim} or incorrect second dim."
        )
        return
    if intrinsic_matrix.shape != (3, 3):
        print(
            f"Error: projection.npy expected a 3x3 intrinsic matrix, got shape {intrinsic_matrix.shape}."
        )
        return

    num_frames, _, H, W = (
        depths.shape
    )  # Unpack dimensions, ignoring the size-1 dimension
    print(f"Detected {num_frames} frames with resolution {W}x{H}.")

    # Extract intrinsic parameters from the 3x3 matrix
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    print(f"Using intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    all_points_3d = []

    # Create pixel coordinate grid once
    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
    u_coords = u_coords.flatten()
    v_coords = v_coords.flatten()

    for i in range(num_frames):
        print(f"Processing frame {i + 1}/{num_frames}...")
        # Access the 2D depth and uncertainty maps by indexing the size-1 dimension
        current_depth_map = depths[i, 0, :, :]
        current_uncertainty_map = uncertainties[i, 0, :, :]
        current_pose_matrix = trajectory[
            i
        ]  # This is the camera-to-world transformation matrix

        # Flatten depth and uncertainty maps for easier iteration
        flat_depths = current_depth_map.flatten()
        flat_uncertainties = current_uncertainty_map.flatten()

        # Filter points based on uncertainty and depth (avoiding zero/negative depths)
        valid_indices = np.where(
            (flat_uncertainties <= uncertainty_threshold) & (flat_depths > 0)
        )[0]

        if len(valid_indices) == 0:
            print(f"  No valid points found in frame {i+1} after filtering.")
            continue

        # Get valid pixel coordinates, depths, and uncertainties
        valid_u = u_coords[valid_indices]
        valid_v = v_coords[valid_indices]
        valid_depths = flat_depths[valid_indices]

        # Unproject all valid pixels in this frame using the extracted intrinsics
        points_camera_coords = np.array(
            [
                unproject_pixel_to_3d(u, v, d, fx, fy, cx, cy)
                for u, v, d in zip(valid_u, valid_v, valid_depths)
            ]
        )

        # Convert to homogeneous coordinates for transformation
        points_camera_homogeneous = np.hstack(
            (points_camera_coords, np.ones((points_camera_coords.shape[0], 1)))
        )

        # Transform points from camera coordinates to world coordinates
        # P_world = T_world_camera @ P_camera
        points_world_homogeneous = (current_pose_matrix @ points_camera_homogeneous.T).T

        # Extract 3D coordinates (discarding the homogeneous component)
        points_world_3d = points_world_homogeneous[:, :3]

        all_points_3d.append(points_world_3d)

    if not all_points_3d:
        print("No points were reconstructed. The output file will not be created.")
        return

    # Concatenate all points into a single NumPy array
    final_point_cloud = np.vstack(all_points_3d)

    np.save(output_filename, final_point_cloud)
    print(f"\nReconstruction complete! Aggregated {final_point_cloud.shape[0]} points.")
    print(f"Point cloud saved to '{output_filename}'")


if __name__ == "__main__":
    # --- IMPORTANT: REPLACE THESE PATHS WITH YOUR ACTUAL FILE PATHS ---
    # Define your actual file paths here:
    your_depths_file = "anycam_output/depths.npy"
    your_trajectory_file = "anycam_output/trajectory.npy"
    your_uncertainties_file = "anycam_output/uncertainties.npy"
    your_projection_file = (
        "anycam_output2/projection.npy"  # This should contain the 3x3 intrinsic matrix
    )

    # Check if dummy files exist for demonstration purposes if actual files are not set
    if not os.path.exists(your_depths_file):
        print("No AnyCam output files found. Creating dummy files for demonstration.")
        num_frames_dummy = 5
        H_dummy, W_dummy = 189, 189  # Use the resolution from your provided shapes

        # Dummy depths and uncertainties with the extra dimension
        dummy_depths = np.random.rand(num_frames_dummy, 1, H_dummy, W_dummy) * 10
        dummy_uncertainties = (
            np.random.rand(num_frames_dummy, 1, H_dummy, W_dummy) * 0.1
        )

        dummy_trajectory = np.array([np.eye(4) for _ in range(num_frames_dummy)])
        for i in range(num_frames_dummy):
            dummy_trajectory[i, 0, 3] = i * 0.1  # Move along X axis
            dummy_trajectory[i, 1, 3] = i * 0.05  # Move along Y axis

        # Dummy intrinsic matrix (example values)
        dummy_intrinsic_matrix = np.array(
            [[300.0, 0.0, W_dummy / 2.0], [0.0, 300.0, H_dummy / 2.0], [0.0, 0.0, 1.0]]
        )

        # Create the directory if it doesn't exist
        os.makedirs("anycam_output2", exist_ok=True)

        np.save(os.path.join("anycam_output2", "depths.npy"), dummy_depths)
        np.save(os.path.join("anycam_output2", "trajectory.npy"), dummy_trajectory)
        np.save(
            os.path.join("anycam_output2", "uncertainties.npy"), dummy_uncertainties
        )
        np.save(
            os.path.join("anycam_output2", "projection.npy"), dummy_intrinsic_matrix
        )

    perform_4d_reconstruction(
        depths_path=your_depths_file,
        trajectory_path=your_trajectory_file,
        uncertainties_path=your_uncertainties_file,
        projection_path=your_projection_file,
        output_filename="anycam_reconstructed_scene.npy",
        uncertainty_threshold=0.05,  # As suggested in the paper
    )

    # Visualization part
    try:
        points = np.load("anycam_reconstructed_scene.npy")

        # Create an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Optional: Estimate normals for better rendering (if you want to render surfaces)
        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Visualize the point cloud
        print("\nDisplaying reconstructed point cloud...")
        o3d.visualization.draw_geometries([pcd])
    except Exception as e:
        print(
            f"Error during visualization: {e}. Make sure Open3D is installed and the .npy file was created."
        )
