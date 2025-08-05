from pydantic import BaseModel
from litserve.mcp import MCP
import litserve as ls
import os
import sys
import subprocess
from typing import Optional


# add system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FloorDataRequest(BaseModel):
    data_folder: str
    output_folder: Optional[str] = None


class FloorDataAPI(ls.LitAPI):
    def setup(self, device):
        REPO_URL = "https://github.com/krash3125/solo-3d.git"
        REPO_PATH = "krash3125_solo_3d"

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

                # Install requirements after cloning
                requirements_path = os.path.join(repo_path, "requirements.txt")
                if os.path.exists(requirements_path):
                    print("Installing requirements from requirements.txt...")
                    try:
                        subprocess.run(
                            ["pip", "install", "-r", requirements_path],
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        print("Requirements installed successfully!")
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to install requirements: {e}")
                        print(f"Error output: {e.stderr}")
                        raise RuntimeError(f"Requirements installation failed: {e}")
                else:
                    print("No requirements.txt found in cloned repository.")

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
            from krash3125_solo_3d.ply_to_2d_map import generate_floor_map

            self.generate_floor_map = generate_floor_map

            print(
                "Successfully imported floor map generation functions from cloned repository!"
            )

        except ImportError as e:
            print(f"Error: Could not import functions from cloned repository: {e}")
            raise e

    def decode_request(self, request: FloorDataRequest):
        return request

    def predict(self, req: FloorDataRequest):
        data_folder = req.data_folder
        output_folder = req.output_folder

        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Data folder not found: {data_folder}")

        if output_folder is None:
            output_folder = data_folder

        data = self.generate_floor_map(data_folder, output_folder)
        return data

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    api = FloorDataAPI(
        mcp=MCP(description="Generates floor maps from PLY point cloud files")
    )
    server = ls.LitServer(api)
    server.run(port=8000)
