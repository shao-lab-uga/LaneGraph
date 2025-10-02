import os
import re
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

def get_files_with_extension(folder_path = '', file_extension = '.parquet'):

    files = []
    for root, _, file in os.walk(folder_path):
        for f in file:
            if f.endswith(file_extension):
                files.append(os.path.join(root, f))

    return files

def get_last_file_with_extension(folder_path = '', file_extension = '.pth'):
    files = []
    
    for root, _, file in os.walk(folder_path):
        for f in file:
            if f.endswith(file_extension):
                files.append(os.path.join(root, f))
    files.sort(key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)) if re.search(r'epoch_(\d+)', x) else -1)
    return files[-1] if len(files) > 0 else None

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
    
def extract_intersection_lat_lon_from_kml(kml_file, save_path=None) -> pd.DataFrame:
    tree = ET.parse(kml_file)
    root = tree.getroot()

    # Define the KML namespace
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    intersection_df = pd.DataFrame()
    latitudes_lst = []
    longitudes_lst = []
    for placemark in root.findall(".//kml:Placemark", ns):
        coordinates = placemark.find('.//kml:Point/kml:coordinates', ns)
        if coordinates is not None:
            # Extract longitude and latitude
            coord_text = coordinates.text.strip()
            coord_parts = coord_text.split(',')
            if len(coord_parts) >= 2:
                longitude = coord_parts[0]
                latitude = coord_parts[1]
                # Write to file
                latitudes_lst.append(float(latitude))
                longitudes_lst.append(float(longitude))

    intersection_df['longitude'] = longitudes_lst
    intersection_df['latitude'] = latitudes_lst
    intersection_df['intersection_id'] = range(len(intersection_df))
    intersection_df['intersection_id'] = intersection_df['intersection_id'].astype(str)
    return intersection_df