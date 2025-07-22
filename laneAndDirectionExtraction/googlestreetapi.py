import requests
import math
from pathlib import Path
"""
 Implement  a bottom half in order to calculate the exact area in which to have
 80m x 80m area
"""

def download_satellite_image(api_key: str, latitude: float, longitude: float):
    """
    Downloads a 640x640 satellite image from Google Maps for the given coordinates.

    The image will represent an area of approximately 150m x 150m. The function
    calculates the appropriate zoom level based on the latitude to achieve this scale.

    Args:
        api_key: Your Google Maps Static API key.
        latitude: The latitude for the center of the image.
        longitude: The longitude for the center of the image.
    """



    try:

        if abs(latitude) > 85:
            print("Warning: Latitude is very close to the pole; zoom calculation may be less accurate.")
            # Use a default high zoom level for polar regions
            zoom = 19
        else:
            required_zoom_float = math.log2((156543.03392 * 640 * math.cos(latitude * math.pi / 180)) / 150)
            zoom = int(round(required_zoom_float))

        print(f"Calculated optimal zoom level: {zoom}")

    except (ValueError, TypeError) as e:
        print(f"Error calculating zoom level: {e}")
        print("Please ensure latitude and longitude are valid numbers.")
        return

    api_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{latitude},{longitude}",
        "zoom": zoom,
        "size": "640x640",
        "maptype": "satellite",
        "key": api_key
    }


    try:
        response = requests.get(api_url, params=params, stream=True)
        response.raise_for_status()
        home_dir = Path.home()
        downloads_path = home_dir / "Downloads"


        downloads_path.mkdir(exist_ok=True)

        file_name = f"satellite_{latitude}_{longitude}.png"
        file_path = downloads_path / file_name

        print(f"Downloading image to: {file_path}")

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


        print(f"Image saved successfully as '{file_name}' in your Downloads folder.")

    except requests.exceptions.RequestException as e:
        print(f"\n An error occurred: {e}")
        print("Please check the following:")




if __name__ == "__main__":

    MY_API_KEY = ""


    target_latitude = 33.95376197266688
    target_longitude = -83.37434831554542
    download_satellite_image(MY_API_KEY, target_latitude, target_longitude)
