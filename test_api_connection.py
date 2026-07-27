import os
from dotenv import load_dotenv
import requests

load_dotenv()

API_KEY = os.getenv("API_KEY")
if API_KEY:
    # Never print the raw key — show only a masked fingerprint for sanity-checking.
    masked = f"{API_KEY[:4]}...{API_KEY[-2:]}" if len(API_KEY) > 6 else "***"
    print(f"API Key: {masked} (length {len(API_KEY)})")
else:
    print("API Key: <not set> — set API_KEY in your environment or .env")

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