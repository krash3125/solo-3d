import torch
import cv2
import numpy as np
import open3d as o3d
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import os
from glob import glob

# === Load HuggingFace Depth Model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "depth-anything/Depth-Anything-V2-Large-hf"

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)

# === Camera intrinsics (adjust if needed) ===
fx, fy = 525.0, 525.0
cx, cy = 319.5, 239.5
width, height = 640, 480

intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# === Load images from folder ===
image_folder = "./img"
image_paths = sorted(glob(os.path.join(image_folder, "*.JPG")))


# === Estimate camera pose from feature matches ===
def estimate_pose(img1, img2):
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 8:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t[:, 0]
    return pose


# === Run DepthAnything ===
def run_depth_estimation(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth.squeeze().cpu().numpy()
    # Resize to match original image size
    depth = cv2.resize(depth, (img_bgr.shape[1], img_bgr.shape[0]))
    return depth


# === Convert RGB + depth + pose to point cloud ===
def create_point_cloud(rgb, depth, pose):
    color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    pcd.transform(pose)
    return pcd


# === Main Loop ===
global_pose = np.eye(4)
global_map = o3d.geometry.PointCloud()

for i in range(len(image_paths)):
    print(f"[{i+1}/{len(image_paths)}] Processing frame: {image_paths[i]}")
    img = cv2.imread(image_paths[i])
    depth = run_depth_estimation(img)

    if i > 0:
        prev_img = cv2.imread(image_paths[i - 1])
        rel_pose = estimate_pose(prev_img, img)
        if rel_pose is not None:
            global_pose = global_pose @ np.linalg.inv(rel_pose)
        else:
            print("Pose estimation failed. Using previous pose.")

    pcd = create_point_cloud(img, depth, global_pose)
    global_map += pcd

# === Final Visualization ===
global_map = global_map.voxel_down_sample(voxel_size=0.02)
global_map.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

o3d.io.write_point_cloud("./reconstruction.ply", global_map)
print("✅ Point cloud saved to reconstruction.ply")

o3d.visualization.draw_geometries([global_map])
