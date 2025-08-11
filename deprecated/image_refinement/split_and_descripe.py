import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import base64
import json

# === Configuration ===
INPUT_IMAGE = "raw_data/sat_0.jpg"  # Replace with your 4096x4096 image
TILE_OUTPUT_DIR = "processed_data/tiles"
JSON_OUTPUT_DIR = "processed_data/tile_prompts"
TILE_SIZE = 1024
GRID_ROWS, GRID_COLS = 4, 4

# === Template instruction block ===
DEFAULT_GOAL_PROMPT = """
Please Edit image, follow the description and goals; Note that the description includes all the road segments, there are no other road segments on the graph.

Goal: Enhance the satellite image to improve road visibility and quality for downstream lane-level road extraction, while strictly preserving all semantic and spatial fidelity. Please follow these instructions exactly:

1. Remove only occlusions without introducing semantic or spatial shifts:
Remove visual obstructions such as trees, vehicles, and shadows that occlude the road surface.
Reconstruct only the occluded portions of existing roads, ensuring smooth and continuous road segments.
Lane positions must not be shifted — retain their exact location to prevent semantic deviation.
Lane markings (e.g., white/yellow lane lines) must be preserved or realistically recovered if occluded.

2. (Important!) Do not hallucinate or alter existing infrastructure:
Do not invent, add, or extend new roads, branches, shortcuts, or driveways.
Do not alter road network connectivity — preserve all topological connections as they appear.
If a road marking (e.g., turn arrows or signs) is ambiguous or completely occluded, do not guess or generate it — omit it instead.

3. Preserve all semantic and geometric features:
Do not modify road geometry, lane alignment, road width, lane markings, traffic signs, intersections, or any infrastructure.
Keep all landmarks, curbs, medians, traffic islands, and barriers unchanged in position, size, and shape (except for removing occlusions).

4. Maintain original spatial scale and resolution:
Do not crop, zoom, resample, upscale, or downscale the image.
The pixel-per-meter scale must remain exactly the same as in the input image.

5. Preserve road and lane dimensional fidelity:
Lane width, length, curvature, and lane count must remain unchanged.
Do not split, merge, narrow, or widen any lanes.

6. Retain original image texture and visual fidelity:
Do not alter the original satellite image texture — preserve all tonal, color, and surface patterns as-is.
Do not apply style-based, artistic, or unrealistic transformations. The enhanced image must appear natural and consistent with high-resolution satellite imagery.

7. Do not modify non-drivable regions:
Do not generate walkways, paths, or markings in vegetated zones, medians, parks, or lawns.
Preserve existing sidewalks, pedestrian areas, and curbs exactly as they are — do not extend, reshape, or modify them.
""".strip()

# === Setup output dirs ===
os.makedirs(TILE_OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

# === Load image ===
img = Image.open(INPUT_IMAGE).convert("RGB")
W, H = img.size
assert W == H == 4096, "Image must be 4096×4096"

# === Split + collect prompt for each tile ===
for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        x0, y0 = col * TILE_SIZE, row * TILE_SIZE
        x1, y1 = x0 + TILE_SIZE, y0 + TILE_SIZE
        tile = img.crop((x0, y0, x1, y1))

        tile_filename = f"tile_{row}_{col}.png"
        tile_path = os.path.join(TILE_OUTPUT_DIR, tile_filename)
        tile.save(tile_path)

        # Show and describe
        plt.imshow(np.array(tile))
        plt.title(f"Tile ({row}, {col})")
        plt.axis("off")
        plt.show(block=False)

        description = input(f"📝 Describe tile ({row}, {col}):\n> ").strip()
        plt.close()

        full_prompt = f"Description: {description}\n\n{DEFAULT_GOAL_PROMPT}"

        # Encode image for GPT API
        with open(tile_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")
        image_url = f"data:image/png;base64,{b64_image}"

        # Construct API-compatible message
        gpt_json = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "input_image", "image_url": image_url}
                    ]
                }
            ]
        }

        json_path = os.path.join(JSON_OUTPUT_DIR, f"tile_{row}_{col}.json")
        with open(json_path, "w") as f:
            json.dump(gpt_json, f, indent=2)

        print(f"✅ JSON saved: {json_path}")
