import requests
import os
import math
import numpy as np
from PIL import Image
from io import BytesIO
def calculate_zoom_for_coverage_google(coverage_meters, image_size=640):
    required_meters_per_pixel = coverage_meters / image_size
    base_resolution = 156543.03392
    zoom = math.log2(base_resolution / required_meters_per_pixel)
    return int(max(0, min(20, round(zoom))))

def download_high_res_google_satellite(lat, lon, folder='satellite_images', 
                                       coverage_meters=80, api_key=None):
    if not api_key:
        print("API key required")
        return None

    zoom = calculate_zoom_for_coverage_google(coverage_meters, 640)

    params = {
        'center': f"{lat},{lon}",
        'zoom': zoom,
        'size': '640x640',
        'maptype': 'satellite',
        'format': 'jpg',
        'scale': 2,
        'key': api_key
    }

    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    os.makedirs(folder, exist_ok=True)
    filename = f"google_satellite_{lat}_{lon}_{coverage_meters}m_640px.jpg"
    filepath = os.path.join(folder, filename)

    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    if len(response.content) > 1000:
        img = Image.open(BytesIO(response.content))
        new_size = (img.width // 2, img.height // 2)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        # Save (either original or downscaled) to disk
        img.save(filepath, quality=100)
        return filepath
    else:
        print("API response error")
        return None

def load_coordinates_from_file(filepath):
    """
    Load coordinates from a text file containing longitude,latitude pairs.
    
    Args:
        filepath (str): Path to the coordinates file
        
    Returns:
        np.ndarray: 2D array with shape (n, 2) where each row is [longitude, latitude]
    """
    try:
        coordinates = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split(',')
                    if len(parts) == 2:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        coordinates.append([lon, lat])
        
        return np.array(coordinates)
    except Exception as e:
        print(f"Error loading coordinates from {filepath}: {e}")
        return None

if __name__ == "__main__":
    GOOGLE_API_KEY = ""

    # Load coordinates from coordinates.txt
    coordinates_file = "coordinates.txt"
    coordinates = load_coordinates_from_file(coordinates_file)
    
    if coordinates is None:
        print(f"Failed to load coordinates from {coordinates_file}")
        exit()
    
    print(f"Loaded {len(coordinates)} coordinate pairs from {coordinates_file}")
    
    if GOOGLE_API_KEY:
        # Iterate through all coordinates
        for i, (lon, lat) in enumerate(coordinates):
            print(f"\nProcessing coordinate {i+1}/{len(coordinates)}: lat={lat}, lon={lon}")
            
            result = download_high_res_google_satellite(
                lat=lat,
                lon=lon,
                folder='raw_data/UGA_Intersections',
                coverage_meters=80,
                api_key=GOOGLE_API_KEY
            )

            if result:
                print(f"Image {i+1} download successful: {result}")
            else:
                print(f"Download failed for coordinate {i+1}")
    else:
        print("No API key provided. Please set GOOGLE_API_KEY to download images.")
        print("Coordinates loaded successfully:")
        for i, (lon, lat) in enumerate(coordinates):
            print(f"  {i+1}: lat={lat}, lon={lon}")