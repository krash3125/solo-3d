from pydantic import BaseModel
from litserve.mcp import MCP
import litserve as ls
import open3d as o3d
import os
import sys
import subprocess

# add system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcloud_to_mesh import (
    load_point_cloud,
    preprocess_point_cloud,
    poisson_reconstruction,
    alpha_shape_reconstruction,
    estimate_normals,
    ball_pivoting_reconstruction,
)


# Load cut3r model
# Generate point cloud --> With specific options


class PointCloudGenerationRequest(BaseModel):
    model_path: str = "src/cut3r_512_dpt_4_64.pth"
    seq_path: str = ""
    size: int = 512
    vis_threshold: float = 1.5
    output_dir: str = "./demo_tmp"
    skip_frames: bool = True


class PointCloudGenerationAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device

        pass

    def decode_request(self, request: PointCloudGenerationRequest):
        return request

    def predict(self, req: PointCloudGenerationRequest):
        model_path = req.model_path
        seq_path = req.seq_path
        device = self.device
        size = req.size
        vis_threshold = req.vis_threshold
        output_dir = req.output_dir
        skip_frames = req.skip_frames

        if device != "cuda":
            raise ValueError("Only CUDA device is supported")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not seq_path or not os.path.exists(seq_path):
            raise FileNotFoundError(f"Sequence directory not found: {seq_path}")

        os.makedirs(output_dir, exist_ok=True)

        # TODO: Implement the actual point cloud generation logic here
        # This would typically involve:
        # 1. Loading the ARCroco3DStereo model
        # 2. Processing the image sequence
        # 3. Generating point clouds
        # 4. Saving results to output_dir

        # Execute the point cloud generation command

        # Build the command with the provided parameters
        cmd = [
            "python",
            "demo_opt.py",
            "--model_path",
            model_path,
            "--size",
            str(size),
            "--seq_path",
            seq_path,
            "--vis_threshold",
            str(vis_threshold),
            "--output_dir",
            output_dir,
            "--device",
            device,
        ]

        # Set environment variable for CUDA memory allocation
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        try:
            # Execute the command
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )

            if result.returncode != 0:
                raise RuntimeError(f"Command failed with error: {result.stderr}")

            print(f"Command output: {result.stdout}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to execute command: {e}")
        except FileNotFoundError:
            raise FileNotFoundError("demo_opt.py not found in the expected location")

        # Placeholder return - replace with actual implementation
        output_path = os.path.join(output_dir, "generated_pointcloud.ply")

        return {
            "output_path": output_path,
            "model_path": model_path,
            "seq_path": seq_path,
            "device": device,
            "size": size,
            "vis_threshold": vis_threshold,
            "skip_frames": skip_frames,
        }

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    pointcloud_api = PointCloudGenerationAPI(
        mcp=MCP(
            description="Generates 3D point clouds from image sequences using ARCroco3DStereo"
        )
    )
    server = ls.LitServer(pointcloud_api)
    server.run(port=8000)
