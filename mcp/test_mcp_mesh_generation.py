import requests

# Example request data
request_data = {
    "ply_path": "./krash3125_solo_3d/pointcloud_to_mesh/fused_filtered_frames.ply",  # Update this path to your test PLY file
    "method": "ball_pivoting",  # Options: 'poisson', 'alpha', 'ball_pivoting'
    "output_path": "./output_mesh.ply",  # Optional
    # "poisson_depth": 8,                         # Optional, for 'poisson'
    # "alpha": 0.03,                              # Optional, for 'alpha'
    # "ball_radii": [0.005, 0.01, 0.02],          # Optional, for 'ball_pivoting'
}

response = requests.post("http://127.0.0.1:8000/predict", json=request_data)

print(f"Status: {response.status_code}\nResponse:\n{response.text}")
