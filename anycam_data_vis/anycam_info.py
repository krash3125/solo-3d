import numpy as np
import open3d as o3d
from PIL import Image
import os
import matplotlib.pyplot as plt

data_dir = "../anycam_output2/"

# Load all data
depths = np.load(os.path.join(data_dir, "depths.npy"))  # (N,1,H,W)
depths = np.squeeze(depths)  # (N,H,W)

poses = np.load(os.path.join(data_dir, "trajectory.npy"))  # (N,4,4)
intrinsics = np.load(os.path.join(data_dir, "projection.npy"))  # (3,3) or (N,3,3)
uncertainties = np.load(os.path.join(data_dir, "uncertainties.npy"))  # (N,1,H,W)
uncertainties = np.squeeze(uncertainties)  # (N,H,W)

frame_files = sorted(
    [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".png")]
)

if intrinsics.ndim == 2:
    intrinsics = np.repeat(intrinsics[None, :, :], len(depths), axis=0)

print(
    f"Data shapes:\n Depths: {depths.shape}\n Poses: {poses.shape}\n Intrinsics: {intrinsics.shape}\n Uncertainties: {uncertainties.shape}\n Frames: {len(frame_files)}"
)

N = len(depths)
print(f"Number of frames: {N}")

# Threshold for uncertainty filtering (adjust as needed)
uncertainty_thresh = 0.1

for i in range(min(N, 5)):
    d = depths[i]
    u = uncertainties[i]
    print(f"Frame {i}: depth min={d.min():.4f}, max={d.max():.4f}")
    print(f"Frame {i}: uncertainty min={u.min():.4f}, max={u.max():.4f}")

    plt.imshow(d, cmap="inferno")
    plt.title(f"Depth Map Frame {i}")
    plt.colorbar()
    plt.show()

    plt.imshow(u, cmap="viridis")
    plt.title(f"Uncertainty Map Frame {i}")
    plt.colorbar()
    plt.show()

    print(f"Frame {i} Pose:\n{poses[i]}")
    print(f"Frame {i} Intrinsics:\n{intrinsics[i]}")

    img = Image.open(frame_files[i])
    print(f"Frame {i} Image size: {img.size}, Depth shape: {d.shape}")

    # Filter valid pixels by depth and uncertainty
    valid_mask = (d > 0) & (u < uncertainty_thresh)
    valid_count = np.sum(valid_mask)
    print(f"Valid pixels count after filtering: {valid_count} out of {d.size}")

    if valid_count == 0:
        print("No valid pixels after filtering; skipping frame.")
        continue

    ys, xs = np.where(valid_mask)

    sample_count = min(100, valid_count)
    sample_indices = np.random.choice(valid_count, size=sample_count, replace=False)
    xs_sample = xs[sample_indices]
    ys_sample = ys[sample_indices]
    zs_sample = d[ys_sample, xs_sample]

    print("Sample depths:", zs_sample[:10])
    print("Sample uncertainties:", u[ys_sample, xs_sample][:10])

    H, W = d.shape
    K = intrinsics[i]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (xs_sample - cx) * zs_sample / fx
    Y = (ys_sample - cy) * zs_sample / fy
    Z = zs_sample

    points_cam = np.vstack((X, Y, Z, np.ones_like(Z)))
    points_world = (poses[i] @ points_cam)[:3]

    print(f"Sample 3D points (world coords) frame {i}:\n", points_world.T[:5])

    # Visualize sampled points as a point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world.T)
    o3d.visualization.draw_geometries([pcd])

print("Diagnostic checks done.")

print(
    """
- If valid pixels count is low, consider increasing uncertainty threshold.
- Check depth scale and intrinsics carefully.
- Make sure depth maps and RGB frames correspond frame-to-frame.
- Use this info to adjust preprocessing before full reconstruction.
"""
)
