import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import open3d as o3d
import os

# ---- Camera Intrinsics (adjust as needed) ----
K = np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]])

# ---- Depth Estimation Pipeline ----
pipe = pipeline(
    task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
)


# ---- Pose Estimation Between Frames ----
def get_pose_from_frames(frame1, frame2, K):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        # Not enough features
        return np.eye(3), np.zeros((3, 1))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 8:
        return np.eye(3), np.zeros((3, 1))
    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        return np.eye(3), np.zeros((3, 1))
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t


# ---- Backproject Depth to 3D Points ----
def depth_to_point_cloud(depth, K, pose, stride=4):
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Subsample pixels
    i, j = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
    z = depth[j, i] / 1000.0  # mm to meters
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    # Remove points with zero depth
    valid = (z > 0).reshape(-1)
    points = points[valid]
    # Transform to world coordinates
    R = pose[:3, :3]
    t = pose[:3, 3]
    points = (R @ points.T + t.reshape(3, 1)).T
    return points


# ---- Main Processing Loop ----
def main():
    video_path = "vid_test.mp4"
    output_ply = "output/merged_point_cloud.ply"
    os.makedirs(os.path.dirname(output_ply), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    poses = [np.eye(4)]
    all_points = []
    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_i > 100:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Estimate pose
        R, t = get_pose_from_frames(prev_gray, gray, K)
        last_pose = poses[-1]
        new_pose = last_pose @ np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
        poses.append(new_pose)
        prev_gray = gray
        # Depth estimation
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        data = pipe(pil_image)
        predicted_depth = data["predicted_depth"]
        depth_np = predicted_depth.cpu().numpy()
        depth_mm = (depth_np * 1000).astype(np.uint16)
        # Backproject to 3D
        points = depth_to_point_cloud(depth_mm, K, new_pose, stride=4)
        all_points.append(points)
        frame_i += 1
        print(f"Processed frame {frame_i}")
    if len(all_points) == 0:
        print("No points to save.")
        return
    all_points = np.concatenate(all_points, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    # Voxel downsample to reduce number of points
    voxel_size = 0.01  # 1cm
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud(output_ply, pcd)
    print(
        f"Saved downsampled point cloud to {output_ply} with {len(pcd.points)} points"
    )


if __name__ == "__main__":
    main()
