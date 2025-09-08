import math
import numpy as np

def lonlat_to_local_m(lons, lats, lat_ref=None):
    """Project lon/lat (deg) to a local meter space using a single scale at lat_ref."""
    if lat_ref is None:
        lat_ref = float(np.mean(lats))
    m_per_deg_lat, m_per_deg_lon = _meters_per_degree(lat_ref)
    x = np.asarray(lons) * m_per_deg_lon
    y = np.asarray(lats) * m_per_deg_lat
    return np.c_[x, y], (m_per_deg_lon, m_per_deg_lat)

def _meters_per_degree(lat_deg: float):
    # simple spheroid approximation
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(lat_deg))
    return m_per_deg_lat, m_per_deg_lon

def convert_pixel_size_to_meter(pixel_size, resolution):
    """
    Convert pixel size to meters at a specific resolution.
    resolution: The resolution in meters per pixel. (e.g. 0.125 pixels/meter)
    """

    return pixel_size * resolution

def calculate_zoom_for_coverage_google(coverage_meters, image_size=640):
    
    required_meters_per_pixel = coverage_meters / image_size
    base_resolution = 156543.03392
    zoom = math.log2(base_resolution / required_meters_per_pixel)
    return int(max(0, min(20, zoom)))

def bbox_from_center_gsd(lat, lon, gsd_m_per_px=0.125, size_px=4096):
    """Make a square bbox (minLon, minLat, maxLon, maxLat) centered at (lat,lon)
       whose width=height = size_px * gsd meters."""
    width_m = size_px * gsd_m_per_px                   # e.g. 4096 * 0.125 = 512 m
    half_m  = width_m / 2.0
    m_per_deg_lat, m_per_deg_lon = _meters_per_degree(lat)
    dlat = half_m / m_per_deg_lat
    dlon = half_m / m_per_deg_lon
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

def dedup_by_poisson(df, min_sep_m=400.0):
    XY, _ = lonlat_to_local_m(df["longitude"].values, df["latitude"].values)
    chosen = []
    used = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        if used[i]:
            continue
        # pick this point
        chosen.append(i)
        # mark points within min_sep_m as used
        dx = XY[:,0] - XY[i,0]; dy = XY[:,1] - XY[i,1]
        used |= (dx*dx + dy*dy) <= (min_sep_m * min_sep_m)
    return df.iloc[chosen].reset_index(drop=True)