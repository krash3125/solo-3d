import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
import glob
import os

# Disable interactive mode
matplotlib.use("Agg")


def generate_floor_map(data_folder: str, output_folder: str):
    # input folder
    pcd_file = "fused.ply"

    # output folder
    os.makedirs(output_folder, exist_ok=True)

    # absolute paths
    data_folder = os.path.abspath(data_folder) + "/"
    output_folder = os.path.abspath(output_folder) + "/"

    # check if data_folder exists
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder {data_folder} does not exist")

    # Read Point Cloud
    pcd = o3d.io.read_point_cloud(data_folder + pcd_file)
    points = np.asarray(pcd.points)

    # Get Min and Max Y
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    mid_way = (min_y + max_y) / 2

    # Filter points below mid_way height
    below_mid_mask = points[:, 1] > mid_way
    below_mid_points = points[below_mid_mask]

    # Create new point cloud with filtered points
    below_mid_pcd = o3d.geometry.PointCloud()
    below_mid_pcd.points = o3d.utility.Vector3dVector(below_mid_points)
    below_mid_pcd.voxel_down_sample(voxel_size=0.09)

    # Create Grid Map
    grid_resolution = 0.02

    x_indices = np.floor(below_mid_points[:, 0] / grid_resolution).astype(int)
    z_indices = np.floor(below_mid_points[:, 2] / grid_resolution).astype(int)

    x_min, z_min = x_indices.min(), z_indices.min()
    x_indices -= x_min
    z_indices -= z_min

    grid_shape = (x_indices.max() + 1, z_indices.max() + 1)
    grid = np.zeros(grid_shape, dtype=int)

    for x, z in zip(x_indices, z_indices):
        grid[x, z] += 1

    # Load Camera Poses and Movement
    camera_folder = data_folder + "camera"

    npz_files = glob.glob(os.path.join(camera_folder, "*.npz"))
    sorted_npz_files = sorted(npz_files)

    # Load each npz file
    npz_data = []
    cam_positions = []

    for npz_file in sorted_npz_files:
        data = np.load(npz_file)
        npz_data.append(data)
        cam_positions.append(data["pose"][:3, 3])
        print(f"Loaded {os.path.basename(npz_file)}")

    cam_positions = np.array(cam_positions)

    cam_x = np.floor(cam_positions[:, 0] / grid_resolution).astype(int) - x_min
    cam_z = np.floor(cam_positions[:, 2] / grid_resolution).astype(int) - z_min
    # Visualize Camera Path

    # Save floor data to a single npz file
    floor_data = {
        "grid": grid,
        "grid_resolution": grid_resolution,
        "x_min": x_min,
        "z_min": z_min,
        "camera_positions": cam_positions,
        "camera_grid_x": cam_x,
        "camera_grid_z": cam_z,
    }

    output_path = os.path.join(output_folder, "floor_data.npz")
    np.savez(output_path, **floor_data)
    print(f"Saved floor data to {output_folder}")

    # THIS IS ALL FOR VISUALIZATION
    # this changes colors to gray
    norm = colors.Normalize(vmin=0, vmax=np.max(grid))
    cmap = colors.LinearSegmentedColormap.from_list(
        "gray_to_black", ["#D3D3D3", "black"]
    )

    plt.ioff()  # Turn off interactive mode
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")

    # add camera x and y graph points
    plt.plot(cam_x, cam_z, color="red", linewidth=2, label="Camera Path", zorder=1)
    plt.scatter(cam_x[0], cam_z[0], color="green", s=25, label="Start", zorder=2)
    plt.scatter(cam_x[-1], cam_z[-1], color="blue", s=25, label="End", zorder=2)

    plt.legend()
    plt.title("2D Grid Map with Camera Trajectory")
    plt.xlabel("X Grid Index")
    plt.ylabel("Z Grid Index")
    plt.grid(False)
    plt.savefig(os.path.join(output_folder, "floor_map.png"))
    plt.close()
    plt.ion()  # Turn interactive mode back on

    return {
        "floor_map_image": os.path.join(output_folder, "floor_map.png"),
        "floor_map_data": os.path.join(output_folder, "floor_data.npz"),
    }
