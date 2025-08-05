import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from matplotlib import colors
import cv2
from ultralytics import YOLO
from IPython.display import display
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
import matplotlib

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

    # Run Object Detection
    model = YOLO("yolo11n.pt")
    frames_folder = data_folder + "color"
    jpg_files = glob.glob(os.path.join(frames_folder, "*.png"))
    sorted_jpg_files = sorted(jpg_files)

    video_path = data_folder + "video.mp4"
    images_to_video(sorted_jpg_files, video_path, 30)
    results = model.track(video_path, show=False, stream=True, tracker="botsort.yaml")

    data_by_frame = get_data_by_frame(results)
    data_by_id = get_data_by_id(data_by_frame)
    data_by_id_cleaned = get_data_by_id_cleaned(data_by_id)

    dirty_annotations = generate_annotations(
        data_by_id_cleaned, data_folder, grid_resolution, x_min, z_min
    )
    annotations = clean_annotations(dirty_annotations, grid_resolution, x_min, z_min)

    # Save floor data to a single npz file
    floor_data = {
        "grid": grid,
        "grid_resolution": grid_resolution,
        "x_min": x_min,
        "z_min": z_min,
        "camera_positions": cam_positions,
        "camera_grid_x": cam_x,
        "camera_grid_z": cam_z,
        "pre_clustered_annotations": dirty_annotations,
        "annotations": annotations,
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

    # add annotations
    for annotation in annotations:
        plt.scatter(
            annotation["grid_point"][0], annotation["grid_point"][1], color="red", s=25
        )
        plt.text(
            annotation["grid_point"][0],
            annotation["grid_point"][1],
            annotation["class"],
            color="red",
            fontsize=10,
        )

    plt.legend()
    plt.title("2D Grid Map")
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


def image_to_world(u, v, depth, K, Tcw):
    """
    u, v: pixel coordinates
    depth: depth at (u, v) in meters
    K: (3x3) camera intrinsics
    Tcw: (4x4) camera-to-world pose
    Returns: (Xw, Yw, Zw) world coordinates
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Camera coordinates
    Xc = (u - cx) * depth / fx
    Yc = (v - cy) * depth / fy
    Zc = depth

    # Homogeneous camera point
    Pc = np.array([Xc, Yc, Zc, 1.0])

    # World coordinates
    Pw = Tcw @ Pc

    return Pw[:3]


def images_to_video(image_paths, output_path, fps=30):
    if not image_paths:
        raise ValueError("No images provided.")

    # Read the first image to get dimensions
    first_image = cv2.imread(image_paths[0])
    height, width, _ = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or use 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Skipping unreadable image: {path}")
            continue
        # Resize if necessary to match dimensions
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        out.write(img)

    out.release()
    print(f"Video saved to {output_path}")


def get_data_by_frame(results):
    data_by_frame = {}
    for frame_i, r in enumerate(results):
        ids = r.boxes.id.tolist() if r.boxes.id is not None else []
        xyxy = r.boxes.xyxy.tolist() if r.boxes.xyxy is not None else []
        names = [r.names[cls.item()] for cls in r.boxes.cls.int()]
        confs = r.boxes.conf.tolist() if r.boxes.conf is not None else []

        data_by_frame[frame_i] = {
            "ids": ids,
            "xyxy": xyxy,
            "names": names,
            "confs": confs,
        }

        print(f"Frame {frame_i}:")
        print(f"Ids: {ids}")
        print(f"Class Names: {names}")
        print(f"Confidence Scores: {confs}")
        print(f"Bounding Box: {xyxy}")
        print("\n")
    return data_by_frame


def get_data_by_id(data_by_frame):
    data_by_id = {}

    for frame_i, data in data_by_frame.items():
        ids = data["ids"]
        names = data["names"]
        confs = data["confs"]
        xyxy = data["xyxy"]

        # For whatever reason some frames may be able to detect objects but no id is assigned so we will just skip over those frames for simplicity purpose rn
        if len(ids) == len(xyxy) == len(names) == len(confs):
            for id, name, conf, xyxy in zip(ids, names, confs, xyxy):
                if id not in data_by_id:
                    data_by_id[id] = {
                        "frames": [],
                        "names": [],
                        "confs": [],
                        "xyxy": [],
                    }
                data_by_id[id]["frames"].append(frame_i)
                data_by_id[id]["names"].append(name)
                data_by_id[id]["confs"].append(conf)
                data_by_id[id]["xyxy"].append(xyxy)

            print(f"Frame {frame_i}:")
            print(f"Ids: {ids}")
            print(f"Class Names: {names}")
            print(f"Confidence Scores: {confs}")
            print(f"Bounding Box: {xyxy}")
            print("\n")
    return data_by_id


def get_data_by_id_cleaned(data_by_id):
    data_by_id_cleaned = {}

    for id, data in data_by_id.items():
        frames = data["frames"]
        names = data["names"]
        confs = data["confs"]
        # bounding box (top-left-x, top-left-y, bottom-right-x, bottom-right-y)
        xyxy = data["xyxy"]

        data_by_id_cleaned[id] = {
            "min_frame": frames[0],
            "max_frame": frames[-1],
            "frames": frames,
            "dominant_class": max(set(names), key=names.count),
            "confidence": confs,
            "average_confidence": sum(confs) / len(confs),
            "center_points": [((p[2] + p[0]) / 2, (p[3] + p[1]) / 2) for p in xyxy],
            "bb": xyxy,
        }
    return data_by_id_cleaned


def generate_annotations(
    data_by_id_cleaned, data_folder, grid_resolution, x_min, z_min
):
    annotations = []
    for key, value in data_by_id_cleaned.items():
        try:
            frames = value["frames"]
            min_frame = value["min_frame"]
            max_frame = value["max_frame"]
            center_point = value["center_points"][0]
            dominant_class = value["dominant_class"]
            confidence = value["confidence"]
            avg_confidence = value["average_confidence"]
            # print(min_frame, max_frame, center_point, dominant_class, confidence)

            frame_id = f"{min_frame:06d}"
            # depth max is opposite (y, x) format
            depth = np.load(data_folder + "depth/" + frame_id + ".npy")[
                round(center_point[1]), round(center_point[0])
            ]
            # depthConf = np.load("./tmp4/conf/" + frame_id + ".npy")[round(center_point[1]), round(center_point[0])]
            camera_data = np.load(data_folder + "camera/" + frame_id + ".npz")
            K = camera_data["intrinsics"]
            pose = camera_data["pose"]

            real_world_point = image_to_world(
                center_point[0], center_point[1], depth, K, pose
            )
            x = real_world_point[0]
            z = real_world_point[2]
            x_normalized = np.floor(x / grid_resolution).astype(int) - x_min
            z_normalized = np.floor(z / grid_resolution).astype(int) - z_min

            annotations.append(
                {
                    "class": dominant_class,
                    "avg_confidence": avg_confidence,
                    "real_world_point": real_world_point,
                    "grid_point": (x_normalized, z_normalized),
                }
            )
        except Exception as e:
            print(e)
            continue

    return annotations


def clean_annotations(annotations, grid_resolution, x_min, z_min):
    # configs
    min_confidence = 0.25
    default_eps = 1.5
    max_eps = 1.5

    # group by class
    class_to_points = defaultdict(list)
    for data in annotations:
        if data["avg_confidence"] < min_confidence:
            continue
        cls = data["class"]
        class_to_points[cls].append(data)

    # calculate eps for each class
    class_to_eps = {}
    for cls, data in class_to_points.items():
        pts = [d["real_world_point"] for d in data]
        pts_arr = np.array(pts)
        if len(pts_arr) < 2:
            # Only one point, assign small eps
            class_to_eps[cls] = 0.5
            continue
        dists = squareform(pdist(pts_arr))
        # Set self-distances to large number to exclude when searching min distance
        np.fill_diagonal(dists, np.inf)
        nearest_dists = dists.min(axis=1)
        # Use median or min distance scaled by a factor (e.g. 1.1) to be slightly more tolerant
        eps_val = max(np.median(nearest_dists), np.min(nearest_dists)) * 1.1

        if eps_val > max_eps:
            class_to_eps[cls] = max_eps
        else:
            class_to_eps[cls] = eps_val

    final_centroids = []

    # Cluster within each class using its custom eps
    for cls, data in class_to_points.items():
        pts = [d["real_world_point"] for d in data]
        pts = np.array(pts)
        eps = class_to_eps.get(cls, default_eps)  # Default to 1.5 if class not in dict
        clustering = DBSCAN(eps=eps, min_samples=1).fit(pts)
        labels = clustering.labels_

        for label in np.unique(labels):
            cluster_pts = pts[labels == label]
            centroid = cluster_pts.mean(axis=0)
            final_centroids.append((cls, tuple(centroid)))  # include class label

    # generate actual grid output
    annotations = []
    for cls, centroid in final_centroids:
        x = centroid[0]
        z = centroid[2]
        x_normalized = np.floor(x / grid_resolution).astype(int) - x_min
        z_normalized = np.floor(z / grid_resolution).astype(int) - z_min
        annotations.append(
            {
                "class": cls,
                "avg_confidence": 1,
                "real_world_point": centroid,
                "grid_point": (x_normalized, z_normalized),
            }
        )
    return annotations
