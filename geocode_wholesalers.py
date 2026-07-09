import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")

INPUT_FILE = os.path.join("data", "wholesalers.csv")
OUTPUT_FILE = os.path.join("data", "wholesalers_geocoded.csv")


def geocode_address(address: str):
    if not MAPBOX_ACCESS_TOKEN:
        raise ValueError("MAPBOX_ACCESS_TOKEN is missing in .env")

    url = "https://api.mapbox.com/geocoding/v5/mapbox.places/" + requests.utils.quote(address) + ".json"

    params = {
        "access_token": MAPBOX_ACCESS_TOKEN,
        "country": "us",
        "limit": 1
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()

    features = data.get("features", [])

    if not features:
        return None, None

    coordinates = features[0]["center"]

    longitude = coordinates[0]
    latitude = coordinates[1]

    return latitude, longitude


def main():
    df = pd.read_csv(INPUT_FILE)

    results = []

    for index, row in df.iterrows():
        name = str(row["name"]).strip()
        address = str(row["address"]).strip()

        print(f"Geocoding {index + 1}/{len(df)}: {name} | {address}")

        try:
            latitude, longitude = geocode_address(address)
        except Exception as e:
            print(f"Failed: {e}")
            latitude, longitude = None, None

        results.append({
            "name": name,
            "address": address,
            "latitude": latitude,
            "longitude": longitude
        })

        time.sleep(0.3)

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)

    print("")
    print(f"Done. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()