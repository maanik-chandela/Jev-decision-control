import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv


# --------------------------------------------------
# Load .env from the project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

print("Loading .env from:", ENV_FILE)
print(".env exists:", ENV_FILE.exists())

load_dotenv(ENV_FILE)

API_KEY = os.getenv("TYPESAFE_API_KEY")

print("API key found:", bool(API_KEY))

if not API_KEY:
    raise RuntimeError("TYPESAFE_API_KEY is missing")


# --------------------------------------------------
# JEV API request
# --------------------------------------------------

url = "https://api.typesafe.ai/v1/systemone"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": "jev-latest",
    "state": "The package was delivered yesterday according to the tracking record.",
    "questions": {
        "delivered": {
            "type": "noul",
            "instructions": "Was the package delivered?",
            "criteria": {
                "true": "The package was delivered.",
                "false": "The package was not delivered."
            }
        }
    }
}


# --------------------------------------------------
# Send request
# --------------------------------------------------

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=60
)

print("\nHTTP status:", response.status_code)

try:
    result = response.json()
    print("\nJEV response:")
    print(json.dumps(result, indent=2))
except ValueError:
    print("\nRaw response:")
    print(response.text)