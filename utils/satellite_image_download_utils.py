import os
import math
import requests
from io import BytesIO
from PIL import Image
from utils.file_utlis import extract_intersection_lat_lon_from_kml
from utils.geo_utils import bbox_from_center_gsd, convert_pixel_size_to_meter, calculate_zoom_for_coverage_google, dedup_by_poisson

def download_google_satellite_image_from_lat_lon(latitude, 
                                                 longitude, 
                                                 save_path='satellite_images', 
                                                 pixel_size=640,
                                                 resolution=0.125,
                                                 api_key=None,
                                                 image_tag=None):
    if not api_key:
        if "GOOGLE_API_KEY" in os.environ:
            api_key = os.environ["GOOGLE_API_KEY"]
            print("Using API key from environment variable")
        else:
            print("API key required")
            raise ValueError("Google API key is required")
    coverage_meters = convert_pixel_size_to_meter(pixel_size, resolution)

    zoom = calculate_zoom_for_coverage_google(coverage_meters, pixel_size)

    params = {
        'center': f"{latitude},{longitude}",
        'zoom': zoom,
        'size': f'{pixel_size}x{pixel_size}',
        'maptype': 'satellite',
        'format': 'jpg',
        'scale': 2,
        'key': api_key
    }

    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    os.makedirs(save_path, exist_ok=True)
    filename = f"google_satellite_{latitude}_{longitude}_{coverage_meters}m_640px.jpg" if image_tag is None else f"{image_tag}.jpg"
    filepath = os.path.join(save_path, filename)

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


def get_mapbox_bbox_stitched(
    bbox,                   # (min_lon, min_lat, max_lon, max_lat)
    out_size=4096,          # final output width=height (px)
    max_req_px=1024,        # per-request cap (<=1280 for Mapbox)
    out_file="mapbox_bbox_4096.jpg",
    token=None
):
    if token is None:
        raise ValueError("You must provide a Mapbox access token")

    min_lon, min_lat, max_lon, max_lat = bbox
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError("Invalid bbox: min_lon < max_lon and min_lat < max_lat required")

    # tile layout
    tiles_x = math.ceil(out_size / max_req_px)
    tiles_y = math.ceil(out_size / max_req_px)
    tile_px_x = min(max_req_px, math.ceil(out_size / tiles_x))
    tile_px_y = min(max_req_px, math.ceil(out_size / tiles_y))

    stitched_w = tiles_x * tile_px_x
    stitched_h = tiles_y * tile_px_y
    mosaic = Image.new("RGB", (stitched_w, stitched_h))

    def lerp(a, b, t): return a + (b - a) * t

    for ix in range(tiles_x):
        x0 = ix * tile_px_x; x1 = x0 + tile_px_x
        fx0 = x0 / stitched_w; fx1 = x1 / stitched_w
        sub_min_lon = lerp(min_lon, max_lon, fx0)
        sub_max_lon = lerp(min_lon, max_lon, fx1)

        for iy in range(tiles_y):
            y0 = iy * tile_px_y; y1 = y0 + tile_px_y
            # y grows down in images → top is max_lat, bottom is min_lat
            fy0 = y0 / stitched_h; fy1 = y1 / stitched_h
            sub_max_lat = lerp(max_lat, min_lat, fy0)  # top
            sub_min_lat = lerp(max_lat, min_lat, fy1)  # bottom

            bbox_str = f"[{sub_min_lon},{sub_min_lat},{sub_max_lon},{sub_max_lat}]"
            BASE_URL = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"
            url = f"{BASE_URL}/{bbox_str}/{tile_px_x}x{tile_px_y}?access_token={token}"

            r = requests.get(url)
            try:
                r.raise_for_status()
            except requests.HTTPError as e:
                # helpful debug print if Mapbox returns 422 / message
                raise RuntimeError(f"Static API error {r.status_code}: {r.text}") from e

            tile_img = Image.open(BytesIO(r.content)).convert("RGB")
            mosaic.paste(tile_img, (x0, y0))

    # crop to exact out_size
    if stitched_w != out_size or stitched_h != out_size:
        mosaic = mosaic.crop((0, 0, out_size, out_size))

    mosaic.save(out_file, "JPEG", quality=95)
    return mosaic


if __name__ == "__main__":
    test_image_non_intersection_lat_lon = (33.818753, -84.352567)
    test_image_intersection_lat_lon = (33.818439, -84.351647)
    os.environ["GOOGLE_API_KEY"] = ""
    lat, lon = test_image_non_intersection_lat_lon
    get_mapbox_bbox_stitched(
        bbox=bbox_from_center_gsd(lat, lon, gsd_m_per_px=0.125, size_px=4096),
        out_size=4096,
        max_req_px=1024,  # safe under Mapbox's 1280 cap
        out_file="test_non_intersection.jpg",
        token=os.environ["MAPBOX_API_TOKEN"]
    )
    result = download_google_satellite_image_from_lat_lon(
            latitude=lat,
            longitude=lon,
            save_path='.',
            image_tag="test_non_intersection",
            pixel_size=640,
            api_key=os.environ["GOOGLE_API_KEY"]
        )
    lat, lon = test_image_intersection_lat_lon
    get_mapbox_bbox_stitched(
        bbox=bbox_from_center_gsd(lat, lon, gsd_m_per_px=0.125, size_px=4096),
        out_size=4096,
        max_req_px=1024,  # safe under Mapbox's 1280 cap
        out_file="test_intersection.jpg",
        token=os.environ["MAPBOX_API_TOKEN"]
    )

    result = download_google_satellite_image_from_lat_lon(
                latitude=lat,
                longitude=lon,
                save_path='.',
                image_tag="test_intersection",
                pixel_size=640,
                api_key=os.environ["GOOGLE_API_KEY"]
            )
    
    RESOLUTION = 0.125  # meters/pixel
    PIXEL_SIZE = 1280  # pixels
    kml_file = "./intersections/kml/intersections.kml"
    satellite_image_save_path = "./intersections/satellite_images"
    intersection_df = extract_intersection_lat_lon_from_kml(kml_file)
    intersection_df = dedup_by_poisson(intersection_df)
    cnt = 0
    for index, row in intersection_df.iterrows():
        get_mapbox_bbox_stitched(bbox=bbox_from_center_gsd(row['latitude'], row['longitude'], gsd_m_per_px=0.125, size_px=4096),
        out_size=4096,
        max_req_px=1024,  # safe under Mapbox's 1280 cap
        out_file=os.path.join(satellite_image_save_path, f"sat_low_res_{str(cnt)}.jpg"),
        token=os.environ["MAPBOX_API_TOKEN"]
        )
        cnt += 1