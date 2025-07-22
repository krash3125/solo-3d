import numpy as np
import open3d as o3d
import cv2
import os

data_dir = "./anycam_output2/"
trajectory = np.load(os.path.join(data_dir, "trajectory.npy"))  # (211,4,4)
depths = np.load(os.path.join(data_dir, "depths.npy"))  # (211,1,189,189)
uncertainties = np.load(os.path.join(data_dir, "uncertainties.npy"))
proj = np.load(os.path.join(data_dir, "projection.npy"))  # (3,3)

frame_files = sorted(
    [f for f in os.listdir(data_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
)

frames = []
for f in frame_files:
    img = cv2.imread(os.path.join(data_dir, f))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    frames.append(img)

flip_z = np.eye(4)
flip_z[2, 2] = -1
fixed_trajectory = [flip_z @ T @ flip_z for T in trajectory]
fixed_trajectory = np.array(fixed_trajectory)

intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(
    width=189, height=189, fx=proj[0, 0], fy=proj[1, 1], cx=proj[0, 2], cy=proj[1, 2]
)

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.004,
    sdf_trunc=0.02,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
)

uncertainty_threshold = 0.3  # soften threshold to weight depth instead of hard mask

for i in range(len(depths)):
    depth = depths[i, 0].copy()  # (189,189)
    uncertainty = uncertainties[i, 0]  # (189,189)

    # Weight depth by confidence (1 - uncertainty)
    depth_weighted = depth * (1.0 - uncertainty)

    # Optional: zero out very uncertain depths (hard mask)
    depth_weighted[uncertainty > uncertainty_threshold] = 0.0

    color = frames[i]

    depth_o3d = o3d.geometry.Image((depth_weighted * 1000).astype(np.uint16))
    color_o3d = o3d.geometry.Image((color * 255).astype(np.uint8))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1000.0,
        depth_trunc=0.5,
        convert_rgb_to_intensity=False,
    )

    extrinsic = np.linalg.inv(fixed_trajectory[i])
    volume.integrate(rgbd, intrinsic, extrinsic)

mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()

# Basic cleanup
mesh.remove_unreferenced_vertices()
mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=150000)

# Save & visualize
o3d.io.write_triangle_mesh("reconstructed_mesh_weighted.ply", mesh)
o3d.visualization.draw_geometries([mesh])
