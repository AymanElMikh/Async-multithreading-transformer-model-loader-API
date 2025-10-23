import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "This is amazing!",
        "model_id": "sentiment"
    }
)

result = response.json()
print(result)