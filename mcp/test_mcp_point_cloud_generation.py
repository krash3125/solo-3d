import requests

request_data = {
    "seq_path": "./001",
}

response = requests.post("http://127.0.0.1:8000/predict", json=request_data)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
