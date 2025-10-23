import requests

response = requests.post(
    "http://localhost:8000/predict/batch",
    json=[
        {"text": "Great product!", "model_id": "sentiment"},
        {"text": "Not satisfied", "model_id": "sentiment"},
        {"text": "It's okay", "model_id": "sentiment"}
    ]
)

results = response.json()

print(results)