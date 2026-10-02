import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TYPESAFE_API_KEY")

if not api_key:
    raise RuntimeError("TYPESAFE_API_KEY is missing")

response = requests.get(
    "https://api.typesafe.ai/v1/models",
    headers={
        "Authorization": f"Bearer {api_key}"
    },
    timeout=30
)

print("HTTP status:", response.status_code)
print(response.text)