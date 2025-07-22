import numpy as np
import open3d as o3d
from PIL import Image
import os
import sys
from tqdm import tqdm

# Check if '-f' flag is passed (include frames)
include_frames = not ("-rf" in sys.argv)

data_dir = "./anycam_output2/"
trajectory = np.load(os.path.join(data_dir, "trajectory.npy"))  # (211, 4, 4)
proj = np.load(os.path.join(data_dir, "projection.npy"))  # (3, 3)
num_frames = trajectory.shape[0]

height, width = 189, 189  # depth resolution

# Construct Open3D camera intrinsics
fx, fy = proj[0, 0], proj[1, 1]
cx, cy = proj[0, 2], proj[1, 2]

# flip matrix
flip_z = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]
)

intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


# === Function to create image plane at a camera pose ===
def create_image_plane(image_path, pose, scale=0.1):
    img = np.array(Image.open(image_path))
    img_o3d = o3d.geometry.Image(img)

    # Create textured quad
    mesh = o3d.geometry.TriangleMesh()
    h, w = img.shape[0], img.shape[1]

    # Plane in camera space: centered at origin, facing forward (z)
    corners = (
        np.array(
            [
                [-w / 2, -h / 2, 0],
                [w / 2, -h / 2, 0],
                [w / 2, h / 2, 0],
                [-w / 2, h / 2, 0],
            ]
        )
        * scale
        / max(w, h)
    )

    # Apply to mesh
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [2, 3, 0]])
    mesh.textures = [img_o3d]
    mesh.triangle_uvs = o3d.utility.Vector2dVector(
        [
            [0, 1],
            [1, 1],
            [1, 0],  # triangle 1
            [1, 0],
            [0, 0],
            [0, 1],  # triangle 2
        ]
    )
    mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])

    # Transform mesh with camera pose
    mesh.transform(pose)
    return mesh


# === Create geometry list ===
geometries = []

for i in range(0, num_frames, 10):  # skip some for speed
    pose = np.linalg.inv(trajectory[i])  # cam-to-world
    corrected_pose = pose @ flip_z

    if include_frames:
        frame_path = os.path.join(data_dir, f"frame_{i}.png")
        if os.path.exists(frame_path):
            mesh = create_image_plane(frame_path, corrected_pose, scale=0.15)
            geometries.append(mesh)

# Add camera coordinate frames regardless of frames flag
for i in range(0, num_frames, 10):
    pose = np.linalg.inv(trajectory[i])  # cam-to-world
    corrected_pose = pose @ flip_z
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    frame.transform(corrected_pose)
    geometries.append(frame)

# === Visualize ===
o3d.visualization.draw_geometries(geometries)
