import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

data_dir = "./anycam_output2/"
depths = np.load(os.path.join(data_dir, "depths.npy"))  # (N,1,H,W)
num_frames = depths.shape[0]


def show_frames_with_depths(frame_indices, cols=3):
    rows = (len(frame_indices) + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))

    for i, idx in enumerate(frame_indices):
        rgb_path = os.path.join(data_dir, f"frame_{idx}.png")
        if not os.path.exists(rgb_path):
            continue

        # Load RGB image and resize to depth resolution
        rgb = Image.open(rgb_path).resize((189, 189), Image.BILINEAR)
        rgb = np.array(rgb)

        depth = depths[idx, 0]
        depth_norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-8)

        plt.subplot(rows, cols * 2, 2 * i + 1)
        plt.imshow(rgb)
        plt.title(f"Frame {idx} RGB (resized)")
        plt.axis("off")

        plt.subplot(rows, cols * 2, 2 * i + 2)
        plt.imshow(depth_norm, cmap="gray")
        plt.title(f"Frame {idx} Depth")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Example: show frames 0, 10, 20, 30, 40, 50 side by side
show_frames_with_depths([0, 10, 20, 30, 40, 50], cols=3)
