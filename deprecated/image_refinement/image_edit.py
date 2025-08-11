from openai import OpenAI
import base64
import os
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)
DEFAULT_GOAL_PROMPT = """
Please Edit image, follow the description and goals; Note that the mask includes all the road segments, there are no other road segments on the figure.

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
test_image = 'raw_data/UGA_intersections/google_satellite_33.9409111101839_-83.37174652434969_80m_640px.jpg'
test_image_mask = 'raw_data/UGA_intersections_masks/google_satellite_33.9409111101839_-83.37174652434969_80m_640px_mask.png'
result = client.images.edit(
    model="gpt-image-1",
    image=open(test_image, "rb"),
    mask=open(test_image_mask, "rb"),
    prompt=DEFAULT_GOAL_PROMPT
)

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Save the image to a file
with open("composition.png", "wb") as f:
    f.write(image_bytes)