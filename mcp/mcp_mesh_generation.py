from pydantic import BaseModel
from litserve.mcp import MCP
import litserve as ls
import open3d as o3d
import os
import sys
import subprocess

# add system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MeshGenerationRequest(BaseModel):
    ply_path: str
    method: str = "poisson"
    output_path: str = None
    poisson_depth: int = 8
    alpha: float = 0.03
    ball_radii: list = [0.005, 0.01, 0.02]


class MeshGenerationAPI(ls.LitAPI):
    def setup(self, device):
        REPO_URL = "https://github.com/krash3125/solo-3d.git"
        REPO_PATH = "~/krash3125-solo-3d"
        # Clone the repository if it doesn't exist
        repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), REPO_PATH)

        if not os.path.exists(repo_path):
            print(f"Cloning repository from {REPO_URL}...")
            try:
                result = subprocess.run(
                    ["git", "clone", REPO_URL, repo_path],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print("Repository cloned successfully!")
            except subprocess.CalledProcessError as e:
                print(f"Failed to clone repository: {e}")
                print(f"Error output: {e.stderr}")
                raise RuntimeError(f"Repository cloning failed: {e}")
        else:
            print("Repository already exists, skipping clone.")

        # Add the cloned repo to Python path
        sys.path.insert(0, repo_path)

        # Import functions from the cloned repository
        try:
            # Import the specific functions you need from the cloned repo
            # Replace these with the actual module and function names from your repo
            from cloned_repo.mesh_utils import load_point_cloud, preprocess_point_cloud
            from cloned_repo.reconstruction import (
                poisson_reconstruction,
                alpha_shape_reconstruction,
            )
            from cloned_repo.normals import estimate_normals
            from cloned_repo.ball_pivoting import ball_pivoting_reconstruction

            # Store the imported functions as instance variables
            self.load_point_cloud = load_point_cloud
            self.preprocess_point_cloud = preprocess_point_cloud
            self.poisson_reconstruction = poisson_reconstruction
            self.alpha_shape_reconstruction = alpha_shape_reconstruction
            self.estimate_normals = estimate_normals
            self.ball_pivoting_reconstruction = ball_pivoting_reconstruction

            print(
                "Successfully imported mesh generation functions from cloned repository!"
            )

        except ImportError as e:
            print(f"Warning: Could not import functions from cloned repository: {e}")
            print("Falling back to local pointcloud_to_mesh module...")
            # Fall back to the local imports if the repo functions aren't available
            from pointcloud_to_mesh import (
                load_point_cloud,
                preprocess_point_cloud,
                poisson_reconstruction,
                alpha_shape_reconstruction,
                estimate_normals,
                ball_pivoting_reconstruction,
            )

            self.load_point_cloud = load_point_cloud
            self.preprocess_point_cloud = preprocess_point_cloud
            self.poisson_reconstruction = poisson_reconstruction
            self.alpha_shape_reconstruction = alpha_shape_reconstruction
            self.estimate_normals = estimate_normals
            self.ball_pivoting_reconstruction = ball_pivoting_reconstruction

        # Store the repo path for later use
        self.repo_path = repo_path

    def decode_request(self, request: MeshGenerationRequest):
        return request

    def predict(self, req: MeshGenerationRequest):
        ply_path = req.ply_path
        method = req.method
        output_path = req.output_path
        poisson_depth = req.poisson_depth
        alpha = req.alpha
        ball_radii = req.ball_radii

        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY file not found: {ply_path}")
        if output_path is None:
            output_path = os.path.splitext(ply_path)[0] + f"_{method}_mesh.ply"

        # Use functions imported from the cloned repository
        pcd = self.load_point_cloud(ply_path)
        mesh = None
        if method == "poisson":
            pcd = self.preprocess_point_cloud(pcd)
            mesh = self.poisson_reconstruction(pcd, depth=poisson_depth)
        elif method == "alpha":
            mesh = self.alpha_shape_reconstruction(pcd, alpha)
        elif method == "ball_pivoting":
            self.estimate_normals(pcd)
            mesh = self.ball_pivoting_reconstruction(pcd, ball_radii)
        else:
            raise ValueError(f"Unknown method: {method}")

        o3d.io.write_triangle_mesh(output_path, mesh)
        return output_path

    def encode_response(self, output):
        return {"mesh_path": output}


if __name__ == "__main__":
    api = MeshGenerationAPI(mcp=MCP(description="Generates meshes from PLY files"))
    server = ls.LitServer(api)
    server.run(port=8000)
