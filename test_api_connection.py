import os
from dotenv import load_dotenv
import requests

load_dotenv()

API_KEY = os.getenv("API_KEY")
print(f"API Key: {API_KEY}")
print(f"API Key length: {len(API_KEY)}")

# Match Postman headers exactly
headers = {
    "x-api-key": API_KEY,
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive"
}

response = requests.get(
    "https://1ywv9dczq5.execute-api.us-east-2.amazonaws.com/ALPBAPI/players",
    headers=headers,
    params={"limit": 3},
    timeout=10
)

print(f"Status: {response.status_code}")
print(f"Response: {response.text[:200]}")