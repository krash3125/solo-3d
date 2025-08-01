from pydantic import BaseModel
from litserve.mcp import MCP
import litserve as ls
import open3d as o3d
import os
import sys
import subprocess
import argparse
from typing import Optional

# add system path
REPO_URL = "https://github.com/krash3125/CUT3R.git"
REPO_PATH = "krash3125_cut3r"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PointCloudGenerationRequest(BaseModel):
    seq_path: str
    model_path: Optional[str] = "src/cut3r_512_dpt_4_64.pth"
    size: Optional[int] = 512
    vis_threshold: Optional[float] = 1.5
    skip_frames: Optional[bool] = True
    output_dir: Optional[str] = ""


class PointCloudGenerationAPI(ls.LitAPI):
    def setup(self, device):
        # Check if CUDA is available
        try:
            import torch

            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available on this system")
        except ImportError:
            raise ImportError("PyTorch is required but not installed")

        success = True
        # Check if running inside Lightning AI
        is_lightning = os.environ.get("LIGHTNING_AI", "false").lower() == "true"

        # Check if conda is installed (only in non-Lightning environments)
        if not is_lightning:
            try:
                result = subprocess.run(
                    ["conda", "--version"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print(f"Conda is installed: {result.stdout.strip()}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Conda is not installed. Installing conda...")

                try:
                    # Download Miniconda installer
                    installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
                    installer_script = "miniconda_installer.sh"

                    # Download the installer
                    subprocess.run(
                        ["wget", installer_url, "-O", installer_script],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    print("Downloaded Miniconda installer")

                    # Make installer executable and run it
                    subprocess.run(["chmod", "+x", installer_script], check=True)

                    # Run installer in batch mode
                    subprocess.run(
                        [
                            "bash",
                            installer_script,
                            "-b",
                            "-p",
                            os.path.expanduser("~/miniconda3"),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    print("Conda installed successfully")

                    # Add conda to PATH
                    conda_path = os.path.expanduser("~/miniconda3/bin")
                    os.environ["PATH"] = f"{conda_path}:{os.environ.get('PATH', '')}"

                    # Initialize conda for current shell
                    subprocess.run(
                        [
                            "bash",
                            "-c",
                            f"source {conda_path}/conda.sh && conda init bash",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )

                    # Clean up installer
                    os.remove(installer_script)
                    print("Conda installation completed")

                    # Re-check conda installation
                    try:
                        result = subprocess.run(
                            ["conda", "--version"],
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        print(f"Conda installation verified: {result.stdout.strip()}")
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        raise RuntimeError(
                            "Conda installation failed or conda not found in PATH"
                        )

                except subprocess.CalledProcessError as e:
                    print(f"Failed to install conda: {e}")
                    print(f"Error output: {e.stderr}")
                    raise RuntimeError(f"Conda installation failed: {e}")
        else:
            print(
                "Lightning AI environment detected - skipping conda installation check"
            )

        # Clone the repository if it doesn't exist
        repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), REPO_PATH)

        marker_path = os.path.join(repo_path, ".cut3r_setup_success")
        if os.path.exists(marker_path):
            print("Setup already completed previously. Skipping.")
            return

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

        # Change to CUT3R directory
        os.chdir(repo_path)
        print(f"Changed to directory: {os.getcwd()}")

        if is_lightning:
            print(
                "Lightning AI environment detected - skipping conda environment creation"
            )
        else:
            # Create conda environment (only in non-Lightning environments)
            print("Creating conda environment 'cut3r'...")
            try:
                subprocess.run(
                    [
                        "conda",
                        "create",
                        "-n",
                        "cut3r",
                        "python=3.11",
                        "cmake=3.14.0",
                        "-y",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print("Conda environment created successfully!")
            except subprocess.CalledProcessError as e:
                print(f"Failed to create conda environment: {e}")
                print(f"Error output: {e.stderr}")

        # Activate conda environment and install dependencies
        print("Installing PyTorch and CUDA dependencies...")
        try:
            # Install PyTorch with CUDA
            cmd = [
                "conda",
                "install",
                "pytorch",
                "torchvision",
                "pytorch-cuda=12.1",
                "-c",
                "pytorch",
                "-c",
                "nvidia",
                "-y",
            ]

            if not is_lightning:
                # Use conda run in non-Lightning environments
                cmd = ["conda", "run", "-n", "cut3r"] + cmd

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            print("PyTorch and CUDA dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install PyTorch dependencies: {e}")
            print(f"Error output: {e.stderr}")
            success = False

        # Install requirements.txt
        print("Installing requirements.txt...")
        try:
            cmd = ["pip", "install", "-r", "requirements.txt"]

            if not is_lightning:
                # Use conda run in non-Lightning environments
                cmd = ["conda", "run", "-n", "cut3r"] + cmd

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            print("Requirements installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install requirements: {e}")
            print(f"Error output: {e.stderr}")
            success = False

        # Install llvm-openmp for PyTorch dataloader issues
        print("Installing llvm-openmp...")
        try:
            cmd = ["conda", "install", "llvm-openmp<16", "-y"]

            if not is_lightning:
                # Use conda run in non-Lightning environments
                cmd = ["conda", "run", "-n", "cut3r"] + cmd

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            print("llvm-openmp installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install llvm-openmp: {e}")
            print(f"Error output: {e.stderr}")
            success = False

        # NOTE: gsplat is not needed if not training
        # # Install gsplat for training logging
        # print("Installing gsplat...")
        # try:
        #     subprocess.run(
        #         [
        #             # TODO: Testing on lightning ai requires this commented becasue u cannot create conda env
        #             # "conda",
        #             # "run",
        #             # "-n",
        #             # "cut3r",
        #             "pip",
        #             "install",
        #             "git+https://github.com/nerfstudio-project/gsplat.git",
        #         ],
        #         check=True,
        #         capture_output=True,
        #         text=True,
        #     )
        #     print("gsplat installed successfully!")
        # except subprocess.CalledProcessError as e:
        #     print(f"Failed to install gsplat: {e}")
        #     print(f"Error output: {e.stderr}")

        # Install evaluation dependencies
        print("Installing evaluation dependencies...")
        try:
            # Install evo
            cmd_evo = ["pip", "install", "evo"]
            if not is_lightning:
                cmd_evo = ["conda", "run", "-n", "cut3r"] + cmd_evo

            subprocess.run(
                cmd_evo,
                check=True,
                capture_output=True,
                text=True,
            )

            # Install open3d
            cmd_open3d = ["pip", "install", "open3d"]
            if not is_lightning:
                cmd_open3d = ["conda", "run", "-n", "cut3r"] + cmd_open3d

            subprocess.run(
                cmd_open3d,
                check=True,
                capture_output=True,
                text=True,
            )
            print("Evaluation dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install evaluation dependencies: {e}")
            print(f"Error output: {e.stderr}")
            success = False

        # Compile CUDA kernels for RoPE
        print("Compiling CUDA kernels for RoPE...")
        try:
            os.chdir("src/croco/models/curope/")
            cmd = ["python", "setup.py", "build_ext", "--inplace"]

            if not is_lightning:
                # Use conda run in non-Lightning environments
                cmd = ["conda", "run", "-n", "cut3r"] + cmd

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            os.chdir("../../../../")
            print("CUDA kernels compiled successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to compile CUDA kernels: {e}")
            print(f"Error output: {e.stderr}")
            success = False
        except FileNotFoundError:
            print("CUDA kernel directory not found, skipping compilation")
            success = False

        print("Downloading gdown...")
        try:
            os.chdir(os.path.join(repo_path))

            cmd = ["pip", "install", "gdown"]
            if not is_lightning:
                cmd = ["conda", "run", "-n", "cut3r"] + cmd

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

            print("Gdown successfully installed!")

        except subprocess.CalledProcessError as e:
            print(f"Failed to download model checkpoints with gdown: {e}")
            print(f"Error output: {e.stderr}")
            success = False

        print("Checking and downloading model checkpoints with gdown...")
        try:
            os.chdir(os.path.join(repo_path, "src"))

            # Define expected model files
            model_files = ["cut3r_512_dpt_4_64.pth", "cut3r_224_linear_4.pth"]

            # Check which files are missing
            missing_files = []
            for file in model_files:
                if not os.path.exists(file):
                    missing_files.append(file)
                    print(f"Model file {file} not found, will download...")
                else:
                    print(f"Model file {file} already exists, skipping download...")

            # Download only missing files
            if missing_files:
                print(f"Downloading {len(missing_files)} missing model files...")

                # First model file (cut3r_512_dpt_4_64.pth)
                if "cut3r_512_dpt_4_64.pth" in missing_files:
                    cmd1 = [
                        "gdown",
                        "--fuzzy",
                        "https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link",
                    ]
                    subprocess.run(
                        cmd1,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    print(
                        "cut3r_512_dpt_4_64.pth model checkpoint downloaded successfully!"
                    )

                # Second model file (cut3r_224_linear_4.pth) - linear model
                if "cut3r_224_linear_4.pth" in missing_files:
                    cmd2 = [
                        "gdown",
                        "--fuzzy",
                        "https://drive.google.com/file/d/11dAgFkWHpaOHsR6iuitlB_v4NFFBrWjy/view?usp=drive_link",
                    ]
                    subprocess.run(
                        cmd2,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    print(
                        "cut3r_224_linear_4.pth (linear) model checkpoint downloaded successfully!"
                    )
            else:
                print("All model files already exist, skipping downloads!")

            # Return to original repo root
            os.chdir(repo_path)
        except subprocess.CalledProcessError as e:
            print(f"Failed to download model checkpoints with gdown: {e}")
            print(f"Error output: {e.stderr}")
            success = False

        # Add the cloned repo to Python path
        sys.path.insert(0, repo_path)

        if success:
            # Write a success marker file to skip setup next time
            try:
                marker_path = os.path.join(repo_path, ".cut3r_setup_success")
                with open(marker_path, "w") as f:
                    f.write("setup_complete\n")
                print(f"Setup completed successfully. Marker written to {marker_path}")
            except Exception as e:
                print(f"Failed to write setup marker: {e}")
        else:
            print("Setup failed. Marker file not written.")

    def decode_request(self, request: PointCloudGenerationRequest):
        return request

    def predict(self, req: PointCloudGenerationRequest):
        model_path = req.model_path
        size = req.size
        vis_threshold = req.vis_threshold
        skip_frames = req.skip_frames
        output_dir = req.output_dir
        seq_path = os.path.abspath(req.seq_path)

        if output_dir == "":
            output_dir = f"{seq_path}_output"
        else:
            output_dir = os.path.abspath(req.output_dir)

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
        ]

        if not skip_frames:
            cmd += ["--skip_frames", "0"]

        # Check if running in Lightning environment
        is_lightning = os.environ.get("LIGHTNING_AI", "false").lower() == "true"

        if not is_lightning:
            cmd = ["conda", "run", "-n", "cut3r"] + cmd

        # Set environment variable for CUDA memory allocation
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        try:
            repo_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), REPO_PATH
            )

            # Execute the command
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                cwd=repo_path,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Command failed with error: {result.stderr}")

            print(f"Command output: {result.stdout}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to execute command: {e}")
        except FileNotFoundError:
            raise FileNotFoundError("demo_opt.py not found in the expected location")

        # Placeholder return - replace with actual implementation
        output_path = os.path.join(output_dir, "fused.ply")

        return {
            "output_folder": output_dir,
            "fused_ply": output_path,
        }

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Point Cloud Generation MCP Server")
    parser.add_argument(
        "--lightning",
        action="store_true",
        help="Flag to indicate running inside Lightning AI",
    )
    args = parser.parse_args()

    # Check if running inside Lightning AI
    is_lightning = args.lightning

    if is_lightning:
        print("Running inside Lightning AI environment")
        # Set environment variable to indicate Lightning AI
        os.environ["LIGHTNING_AI"] = "true"
    else:
        print("Running in standard environment")

    pointcloud_api = PointCloudGenerationAPI(
        mcp=MCP(
            description="Generates 3D point clouds from image sequences using ARCroco3DStereo"
        )
    )
    server = ls.LitServer(pointcloud_api)
    server.run(port=8000)
