import requests

request_data = {
    "data_folder": "./001_output/",
}

response = requests.post("http://127.0.0.1:8000/predict", json=request_data)

print(f"Status: {response.status_code}\nResponse:\n{response.text}")
