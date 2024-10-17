import requests

# Replace with your actual endpoint URL and API token
API_URL = "https://xw17frz89y9teak9.us-east-1.aws.endpoints.huggingface.cloud"
API_TOKEN = "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"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# Replace with appropriate input for your model
data = {
    "inputs": "The film was a thrilling adventure with stunning visuals."
}

response = requests.post(API_URL, headers=headers, json=data)

# Check if the request was successful
if response.status_code == 200:
    print("Inference successful!")
    print("Model Output:", response.json())
else:
    print(f"Request failed with status code {response.status_code}")
    print("Error:", response.text)