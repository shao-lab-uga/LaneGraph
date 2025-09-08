import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import cv2
import json
import numpy as np
from time import time
from roadstructure import LaneMap

import math
import json
from typing import List, Tuple

LITE_RENDER = False

# -------------- NEW: image/dataset helpers --------------
### NEW: allow directory or single image
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(path: str) -> List[str]:
    if os.path.isdir(path):
        files = sorted(
            [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.splitext(f.lower())[1] in IMG_EXTS
            ]
        )
        if not files:
            raise FileNotFoundError(f"No images found in directory: {path}")
        return files
    else:
        if os.path.splitext(path.lower())[1] not in IMG_EXTS:
            raise ValueError(f"Unsupported image type: {path}")
        return [path]

def default_annotation_path(img_path: str) -> str:
    root, ext = os.path.splitext(img_path)
    # switch to JSON instead of pickle
    return f"{root}_label.json"


def load_lane_maps(ann_path: str) -> Tuple[List[LaneMap], int]:
    """
    Load annotations from JSON and ensure we always have exactly 3 layers:
      layer 0: graph
      layer 1: mask (red)
      layer 2: mask (blue)
    If file is missing or malformed, fall back to empty layers.
    """
    lane_maps: List[LaneMap] = []

    if os.path.exists(ann_path):
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                # list of lane maps serialized via to_dict()
                for item in data:
                    if isinstance(item, dict):
                        lane_maps.append(LaneMap.from_dict(item))
            elif isinstance(data, dict):
                # single map saved earlier → keep behavior from your old code
                lane_maps.append(LaneMap.from_dict(data))
            else:
                # unknown format → start fresh
                lane_maps = []
        except Exception as e:
            print(f"[WARN] Failed to load JSON annotations '{ann_path}': {e}")
            lane_maps = []
    else:
        lane_maps = []

    # ensure we have exactly 3 layers total
    while len(lane_maps) < 3:
        lane_maps.append(LaneMap())
    if len(lane_maps) > 3:
        lane_maps = lane_maps[:3]

    # consistency checks for all layers
    for lm in lane_maps:
        lm.checkConsistency()
        lm.updateNodeType()

    active = 0
    return lane_maps, active


def save_lane_maps(ann_path: str, lane_maps: List[LaneMap]):
    """
    Save the three LaneMaps as a JSON array of dicts.
    """
    payload = [lm.to_dict(include_history=False) for lm in lane_maps]
    tmp_path = ann_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, ann_path)

# -------------- load image list --------------
if len(sys.argv) < 2:
    print("Usage: python editor.py <image_or_directory> [annotation_path(optional)] [config.json(optional)] [swap_bgr(optional)]")
    sys.exit(1)

input_path = sys.argv[1]
image_paths = list_images(input_path)
img_idx = 0

# If user supplied explicit annotation path AND single image, keep using it.
explicit_annotation = None
if len(sys.argv) > 2 and os.path.isfile(sys.argv[2]) and len(image_paths) == 1:
    explicit_annotation = sys.argv[2]

# -------------- globals that change per image --------------
image = None
image_for_cnn = None
annotation = None
laneMaps = None
laneMap = None
activeLaneMap = 0  # 0=graph, 1=mask(red), 2=mask(blue)

# -------------- editor state --------------
margin = 0
margin_for_cnn = 0

# Slightly wider window to hold a right sidebar (cheat-sheet)
SIDEBAR_W = 300                 # ### NEW: sidebar width
windowsize = [1280 + SIDEBAR_W, 1280]  # ### NEW: add sidebar space
VIEW_W = 1280                    # drawing area width (left)
VIEW_H = 1280

pos = [0, 0]
minimap = None
renderTime = 0

vis_switch_no_direction = False
vis_switch_no_arrow = False
vis_switch_no_minimap = False
vis_switch_no_diff_way_and_link = False
vis_switch_no_vertices = False
vis_switch_no_way = False
vis_switch_no_link = False

step = 0
lastMousex, lastMousey = -1, -1
mousex, mousey = 0, 0
lastNodeID = None
deleteMode = " "

editingMode = "ready_to_draw"
autofill_mode = "bezier"
autofill_nodes = [-1, -1]
edgeType = "way"
activeLinks = []
zoom = 1
erase_size = 1

# mask layer colors
MASK_COLORS = {
    1: (0, 0, 255),    # red for layer 1
    2: (255, 0, 0)     # blue for layer 2
}

# -------------- image/color utils --------------
def load_image_and_init(path: str):
    """Load image, color swap if requested, pad, (re)build minimap and globals."""
    global image, image_for_cnn, minimap, dim, imageDim, pos

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")

    # Optional BGR<->RGB swap if extra args present (kept your original behavior)
    if len(sys.argv) > 4:
        b, g, r = np.copy(img[:, :, 0]), np.copy(img[:, :, 1]), np.copy(img[:, :, 2])
        img[:, :, 0] = r
        img[:, :, 1] = b
        img[:, :, 2] = g

    img = np.pad(img, [[margin, margin], [margin, margin], [0, 0]], "constant")
    image_for_cnn = np.pad(
        img,
        [
            [margin_for_cnn - margin, margin_for_cnn - margin],
            [margin_for_cnn - margin, margin_for_cnn - margin],
            [0, 0],
        ],
        "constant",
    )

    image = img
    dim = np.shape(image)
    imageDim = dim
    pos = [0, 0]  # reset view when switching images
    # Build minimap of base image (no overlay)
    build_minimap()

def build_minimap():
    """Rebuild minimap from the base image."""
    global minimap
    minimap = cv2.resize(image, (256, 256))

def current_annotation_path() -> str:
    if explicit_annotation is not None:
        return explicit_annotation
    return default_annotation_path(image_paths[img_idx])

def load_annotations_for_current():
    global laneMaps, activeLaneMap, laneMap
    ann = current_annotation_path()
    laneMaps, activeLaneMap = load_lane_maps(ann)
    laneMap = laneMaps[activeLaneMap]

def save_annotations_for_current():
    ann = current_annotation_path()
    save_lane_maps(ann, laneMaps)

# -------------- UI: sidebar cheat-sheet --------------
### NEW: Draw a right-side help panel with key bindings & image info
SIDEBAR_PAD = 12
LINE_H = 22
FONT = cv2.FONT_HERSHEY_SIMPLEX

SHORTCUTS = [
    ("Pan", "w a s d"),
    ("Zoom", "1 2 3"),
    ("Edit/Draw", "e"),
    ("Link/Way", "q"),
    ("Delete toggle", "x"),
    ("Erase toggle", "r"),
    ("Eraser size", "4 / 5 / 6"),
    ("Undo", "z"),
    ("Layer", "m"),
    ("Bezier (turning)", "f"),
    ("Select turn lane", "c"),
    ("Toggle minimap", "t"),
    ("Prev/Next image", "p / n"),
    ("Exit", "ESC"),
]

def draw_sidebar(frame, info_lines=None):
    # frame is VIEW sized (640x640); we compose a full window by placing sidebar next to it.
    sidebar = np.zeros((VIEW_H, SIDEBAR_W, 3), dtype=np.uint8)
    sidebar[:] = (28, 28, 28)  # dark background

    y = SIDEBAR_PAD + 5
    def put(text, y, scale=0.5, color=(230, 230, 230), thick=1):
        cv2.putText(sidebar, text, (SIDEBAR_PAD, y), FONT, scale, color, thick, cv2.LINE_AA)

    put("Image Editor", y, 0.7, (255, 255, 255), 2); y += LINE_H + 8
    # status
    cur = img_idx + 1
    tot = len(image_paths)
    put(f"Image: {cur}/{tot}", y); y += LINE_H
    put(os.path.basename(image_paths[img_idx])[:36], y); y += LINE_H

    # modes
    put("Modes:", y, 0.6, (180, 220, 255), 2); y += LINE_H
    put(f"  edit: {editingMode}{deleteMode}", y); y += LINE_H
    put(f"  type: {edgeType}", y); y += LINE_H
    put(f"  layer: {activeLaneMap}", y); y += LINE_H
    put(f"  zoom: {zoom}x", y); y += LINE_H
    put(f"  eraser: {erase_size}", y); y += LINE_H

    y += 6
    put("Shortcuts", y, 0.6, (180, 220, 255), 2); y += LINE_H
    for k, v in SHORTCUTS:
        put(f"- {k}: {v}", y); y += LINE_H

    if info_lines:
        y += 8
        put("Info:", y, 0.6, (180, 220, 255), 2); y += LINE_H
        for line in info_lines:
            put(line[:36], y); y += LINE_H

    # compose right next to frame
    full = np.zeros((VIEW_H, VIEW_W + SIDEBAR_W, 3), dtype=np.uint8)
    full[:, :VIEW_W] = frame
    full[:, VIEW_W:] = sidebar
    return full

# -------------- Mouse Handler --------------
def mouseEventHandler(event, x, y, flags, param):
    global mousex, mousey
    global editingMode, edgeType, lastNodeID, activeLinks, zoom, laneMap, erase_size
    mousex, mousey = x, y

    # Clamp mouse to drawing area (ignore sidebar)
    if mousex >= VIEW_W:  # in sidebar
        return

    global_x, global_y = x // zoom + pos[0], y // zoom + pos[1]

    if event == cv2.EVENT_LBUTTONUP:
        changed = False
        if editingMode == "ready_to_draw":
            existing_node = laneMap.query((global_x, global_y), activeLinks)
            if existing_node is None:
                lastNodeID = laneMap.addNode((global_x, global_y))
                editingMode = "drawing_polyline"
                changed = True
            else:
                lastNodeID = existing_node
                editingMode = "drawing_polyline"

        elif editingMode == "drawing_polyline":
            existing_node = laneMap.query((global_x, global_y), activeLinks)
            if existing_node is None:
                if deleteMode == "(delete)":
                    pass
                else:
                    nid = laneMap.addNode((global_x, global_y))
                    laneMap.addEdge(lastNodeID, nid, edgeType)
                    lastNodeID = nid
                    changed = True
            else:
                if existing_node == lastNodeID:
                    if deleteMode == "(delete)":
                        laneMap.deleteNode(lastNodeID)
                        if lastNodeID in activeLinks:
                            activeLinks.remove(lastNodeID)
                        changed = True
                    editingMode = "ready_to_draw"
                    lastNodeID = None
                else:
                    if deleteMode == "(delete)":
                        laneMap.deleteEdge(lastNodeID, existing_node)
                        changed = True
                    else:
                        laneMap.addEdge(lastNodeID, existing_node, edgeType)
                        changed = True
                    lastNodeID = None
                    editingMode = "ready_to_draw"

        elif editingMode == "ready_to_edit":
            existing_node = laneMap.query((global_x, global_y), activeLinks)
            if existing_node is not None:
                lastNodeID = existing_node
                editingMode = "editing"

        elif editingMode == "editing":
            editingMode = "ready_to_edit"
            lastNodeID = None

        elif editingMode == "selecting_link":
            existing_node = laneMap.query((global_x, global_y))
            if existing_node is not None:
                activeLinks = laneMap.findLink(existing_node)
            else:
                activeLinks = []
            editingMode = "ready_to_edit"

        elif editingMode == "autofill_stage1":
            existing_node = laneMap.query((global_x, global_y))
            if existing_node is not None:
                autofill_nodes[0] = existing_node
                editingMode = "autofill_stage2"
                lastNodeID = existing_node

        elif editingMode == "autofill_stage2":
            existing_node = laneMap.query((global_x, global_y))
            if existing_node is not None:
                autofill_nodes[1] = existing_node
                if autofill_mode == "bezier":
                    editingMode = "autofill_stage3"
                    lastNodeID = None
                else:
                    # keep your ML branch as-is if you enable it later
                    editingMode = "autofill_stage1"
                    lastNodeID = None

        elif editingMode == "autofill_stage3":
            x1, y1 = laneMap.nodes[autofill_nodes[0]]
            x2, y2 = laneMap.nodes[autofill_nodes[1]]
            x3, y3 = global_x, global_y
            L = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            N = int(L / 20) + 1

            def interpolate(p1, p2, a):
                return (p1[0] * (1 - a) + p2[0] * a, p1[1] * (1 - a) + p2[1] * a)

            prev_nid = autofill_nodes[0]
            for i in range(N):
                alpha = float(i + 1) / N
                loc = interpolate(
                    interpolate((x1, y1), (x3, y3), alpha),
                    interpolate((x3, y3), (x2, y2), alpha),
                    alpha,
                )
                if i == N - 1:
                    nid = autofill_nodes[1]
                else:
                    nid = laneMap.addNode((int(loc[0]), int(loc[1])))
                laneMap.addEdge(prev_nid, nid, edgetype="link")
                prev_nid = nid
            editingMode = "autofill_stage1"
            changed = True

        elif editingMode == "erase":
            rmlist = []
            for nid, loc in laneMap.nodes.items():
                if (
                    loc[0] > global_x - erase_size * 50 // zoom
                    and loc[0] < global_x + erase_size * 50 // zoom
                    and loc[1] > global_y - erase_size * 50 // zoom
                    and loc[1] < global_y + erase_size * 50 // zoom
                ):
                    rmlist.append(nid)
            for nid in rmlist:
                laneMap.deleteNode(nid)
            changed = True

        if changed:
            save_annotations_for_current()

    if editingMode == "editing":
        laneMap.nodes[lastNodeID] = [global_x, global_y]

    redraw()

def dashline(img, p1, p2, color, width, linetype):
    L = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    w = 8
    for i in range(-w // 2, int(L), w * 2):
        a1 = max(0, float(i) / L)
        a2 = float(i + w) / L
        x1 = int(p1[0] * (1 - a1) + p2[0] * a1); y1 = int(p1[1] * (1 - a1) + p2[1] * a1)
        x2 = int(p1[0] * (1 - a2) + p2[0] * a2); y2 = int(p1[1] * (1 - a2) + p2[1] * a2)
        cv2.line(img, (x1, y1), (x2, y2), color, width, linetype)

def _render_graph_layer(frame):
    """Original LAYER-0 (graph) rendering, isolated so other layers never leak."""
    global lastNodeID
    curLM = laneMaps[0]
    currentLastNodeID = lastNodeID  # preserve after this block

    # main edge rendering (two-pass shadows)
    for renderpass in [0, 1]:
        for nid, nei in curLM.neighbors.items():
            x1 = (curLM.nodes[nid][0] - pos[0]) * zoom
            y1 = (curLM.nodes[nid][1] - pos[1]) * zoom
            outrange = (x1 < 0 or x1 > VIEW_W) or (y1 < 0 or y1 > VIEW_H)
            for nn in nei:
                lanetype = "way" if curLM.edgeType[(nid, nn)] == "way" else "link"
                x2 = (curLM.nodes[nn][0] - pos[0]) * zoom
                y2 = (curLM.nodes[nn][1] - pos[1]) * zoom
                if (x2 < 0 or x2 > VIEW_W) or (y2 < 0 or y2 > VIEW_H):
                    if outrange:
                        continue

                dx, dy = x2 - x1, y2 - y1
                l = math.sqrt(float(dx * dx + dy * dy)) + 1e-3
                dx, dy = dx / l, dy / l

                if vis_switch_no_direction:
                    if lanetype == "way": dx, dy = 1, 0
                    else: dx, dy = 0, 1

                if lanetype == "way":
                    color = (192, 192 + int(dx * 63), 192 + int(dy * 63))
                else:
                    color = (127, 127 + int(dx * 127), 127 + int(dy * 127))

                if vis_switch_no_diff_way_and_link:
                    color = (192, 255, 192)
                if vis_switch_no_link and lanetype == "link":
                    continue
                if vis_switch_no_way and lanetype == "way":
                    continue

                scale = 6
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                ms = []
                arrow_int = 40
                if l > arrow_int:
                    N = int(l / arrow_int)
                    for k in range(N + 1):
                        a = float(k + 1) / (N + 1)
                        ms.append((int(x1 * (1 - a) + x2 * a), int(y1 * (1 - a) + y2 * a)))
                else:
                    ms = [(int(mx), int(my))]

                shadow_color = (96, 96, 96)
                shadow_width = 4

                if lanetype == "link":
                    if LITE_RENDER or vis_switch_no_direction:
                        dashline(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)
                    else:
                        if renderpass == 0:
                            for mx, my in ms:
                                if not vis_switch_no_arrow:
                                    cv2.line(frame, (mx + int(dx * scale), my + int(dy * scale)),
                                             (mx - int(dy * scale), my + int(dx * scale)),
                                             shadow_color, shadow_width, cv2.LINE_AA)
                                    cv2.line(frame, (mx + int(dx * scale), my + int(dy * scale)),
                                             (mx + int(dy * scale), my - int(dx * scale)),
                                             shadow_color, shadow_width, cv2.LINE_AA)
                            dashline(frame, (int(x1), int(y1)), (int(x2), int(y2)), shadow_color, shadow_width, cv2.LINE_AA)
                        else:
                            for mx, my in ms:
                                if not vis_switch_no_arrow:
                                    cv2.line(frame, (mx + int(dx * scale), my + int(dy * scale)),
                                             (mx - int(dy * scale), my + int(dx * scale)),
                                             color, 2)
                                    cv2.line(frame, (mx + int(dx * scale), my + int(dy * scale)),
                                             (mx + int(dy * scale), my - int(dx * scale)),
                                             color, 2)
                            dashline(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)
                else:
                    if LITE_RENDER or vis_switch_no_direction:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)
                    else:
                        if renderpass == 0:
                            for mx, my in ms:
                                if not vis_switch_no_arrow:
                                    cv2.line(frame, (mx + int(dx * scale), my + int(dy * scale)),
                                             (mx - int(dy * scale), my + int(dx * scale)),
                                             shadow_color, shadow_width, cv2.LINE_AA)
                                    cv2.line(frame, (mx + int(dx * scale), my + int(dy * scale)),
                                             (mx + int(dy * scale), my - int(dx * scale)),
                                             shadow_color, shadow_width, cv2.LINE_AA)
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), shadow_color, shadow_width, cv2.LINE_AA)
                        else:
                            for mx, my in ms:
                                if not vis_switch_no_arrow:
                                    cv2.line(frame, (mx + int(dx * scale), my + int(dy * scale)),
                                             (mx - int(dy * scale), my + int(dx * scale)),
                                             color, 2)
                                    cv2.line(frame, (mx + int(dx * scale), my + int(dy * scale)),
                                             (mx + int(dy * scale), my - int(dx * scale)),
                                             color, 2)
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)

    # vertices
    if not vis_switch_no_vertices:
        for nid, p in curLM.nodes.items():
            x1 = (p[0] - pos[0]) * zoom; y1 = (p[1] - pos[1]) * zoom
            if 0 <= x1 < VIEW_W and 0 <= y1 < VIEW_H:
                if curLM.nodeType[nid] == "way" and not vis_switch_no_way:
                    cv2.circle(frame, (int(x1), int(y1)), 4, (0, 0, 0), -1)
                elif curLM.nodeType[nid] == "link" and not vis_switch_no_link:
                    cv2.circle(frame, (int(x1), int(y1)), 3, (255, 0, 0), -1)

    # highlight way nodes adjacent to any link
    if not vis_switch_no_way:
        for nid, p in curLM.nodes.items():
            x1 = (p[0] - pos[0]) * zoom; y1 = (p[1] - pos[1]) * zoom
            if 0 <= x1 < VIEW_W and 0 <= y1 < VIEW_H:
                if curLM.nodeType[nid] == "way":
                    for nei in curLM.neighbors_all[nid]:
                        if (((nei, nid) in curLM.edgeType and curLM.edgeType[(nei, nid)] == "link") or
                            ((nid, nei) in curLM.edgeType and curLM.edgeType[(nid, nei)] == "link")):
                            cv2.circle(frame, (int(x1), int(y1)), 5, (255, 0, 0), 2)

    # active link highlight
    for nid in activeLinks:
        x1 = (curLM.nodes[nid][0] - pos[0]) * zoom; y1 = (curLM.nodes[nid][1] - pos[1]) * zoom
        if 0 <= x1 < VIEW_W and 0 <= y1 < VIEW_H:
            cv2.circle(frame, (int(x1), int(y1)), 5, (255, 255, 0), 5, 1)

    # hover highlight
    global_x, global_y = min(mousex, VIEW_W - 1) // zoom + pos[0], min(mousey, VIEW_H - 1) // zoom + pos[1]
    existing_node = curLM.query((global_x, global_y), activeLinks)
    if existing_node is not None:
        x1 = (curLM.nodes[existing_node][0] - pos[0]) * zoom; y1 = (curLM.nodes[existing_node][1] - pos[1]) * zoom
        cv2.circle(frame, (int(x1), int(y1)), 5, (0, 0, 255), 2)

    # preview line while drawing
    if (editingMode in ["drawing_polyline", "autofill_stage2"]) and lastNodeID is not None:
        x1 = (curLM.nodes[lastNodeID][0] - pos[0]) * zoom; y1 = (curLM.nodes[lastNodeID][1] - pos[1]) * zoom
        x2 = min(mousex, VIEW_W - 1); y2 = min(mousey, VIEW_H - 1)
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                 (0, 255, 0) if edgeType == "way" else (255, 0, 0), 2, cv2.LINE_AA)

    # restore lastNodeID
    lastNodeID = currentLastNodeID
    return frame

def _render_mask_layer(frame, layer_idx):
    """Mask/polygon rendering for layers 1 and 2."""
    global lastNodeID
    color = MASK_COLORS.get(layer_idx, (0, 0, 255))
    maskLM = laneMaps[layer_idx]
    currentLastNodeID = lastNodeID  # preserve after this block

    mask = np.zeros_like(frame)

    # polygons
    polygons = maskLM.findAllPolygons()
    for polygon in polygons:
        polygon_list = []
        for i in range(len(polygon) - 1):
            x1 = (maskLM.nodes[polygon[i]][0] - pos[0]) * zoom
            y1 = (maskLM.nodes[polygon[i]][1] - pos[1]) * zoom
            x2 = (maskLM.nodes[polygon[i + 1]][0] - pos[0]) * zoom
            y2 = (maskLM.nodes[polygon[i + 1]][1] - pos[1]) * zoom
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)
            polygon_list.append([int(x1), int(y1)])
        if len(polygon) >= 2:
            polygon_list.append([int(x2), int(y2)])

        if len(polygon_list) >= 3:
            area = np.array(polygon_list, dtype=np.int32)
            cv2.fillPoly(mask, [area], tuple(int(c*0.2)+50 for c in color))  # softer fill

    # link-style arrows on edges
    for nid, nei in maskLM.neighbors.items():
        x1 = (maskLM.nodes[nid][0] - pos[0]) * zoom; y1 = (maskLM.nodes[nid][1] - pos[1]) * zoom
        for nn in nei:
            x2 = (maskLM.nodes[nn][0] - pos[0]) * zoom; y2 = (maskLM.nodes[nn][1] - pos[1]) * zoom
            dx, dy = x2 - x1, y2 - y1
            l = math.sqrt(float(dx * dx + dy * dy)) + 1e-3
            dx, dy = dx / l, dy / l
            scale = 5
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.line(mask, (int(mx + dx * scale), int(my + dy * scale)),
                     (int(mx - dy * scale), int(my + dx * scale)), color, 2)
            cv2.line(mask, (int(mx + dx * scale), int(my + dy * scale)),
                     (int(mx + dy * scale), int(my - dx * scale)), color, 2)
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)

    # nodes
    for nid, p in maskLM.nodes.items():
        x1 = (p[0] - pos[0]) * zoom; y1 = (p[1] - pos[1]) * zoom
        cv2.circle(mask, (int(x1), int(y1)), 3, (0, 255, 255), -1)

    # hover highlight
    global_x, global_y = min(mousex, VIEW_W - 1) // zoom + pos[0], min(mousey, VIEW_H - 1) // zoom + pos[1]
    existing_node = maskLM.query((global_x, global_y))
    if existing_node is not None:
        x1 = (maskLM.nodes[existing_node][0] - pos[0]) * zoom; y1 = (maskLM.nodes[existing_node][1] - pos[1]) * zoom
        cv2.circle(mask, (int(x1), int(y1)), 5, (0, 0, 255), 2)

    # preview line while drawing
    if editingMode == "drawing_polyline" and lastNodeID is not None:
        x1 = (maskLM.nodes[lastNodeID][0] - pos[0]) * zoom; y1 = (maskLM.nodes[lastNodeID][1] - pos[1]) * zoom
        x2 = min(mousex, VIEW_W - 1); y2 = min(mousey, VIEW_H - 1)
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)),
                 (0, 255, 0) if edgeType == "way" else (255, 0, 0), 2, cv2.LINE_AA)

    frame = cv2.add(frame, mask)

    # restore lastNodeID
    lastNodeID = currentLastNodeID
    return frame

def redraw(noshow=False, transpose=False):
    global zoom, activeLinks, laneMap, activeLaneMap, lastNodeID, renderTime
    global vis_switch_no_direction, vis_switch_no_minimap, vis_switch_no_diff_way_and_link
    global vis_switch_no_vertices, vis_switch_no_way, vis_switch_no_link

    t0 = time()

    # draw region is fixed to VIEW_W×VIEW_H
    base = np.copy(
        image[
            pos[1] : pos[1] + VIEW_H // zoom,
            pos[0] : pos[0] + VIEW_W // zoom,
            :,
        ]
    )
    frame = cv2.resize(base, (VIEW_W, VIEW_H)) if zoom > 1 else base.copy()

    # === IMPORTANT: only render the ACTIVE layer ===
    if activeLaneMap == 0:
        frame = _render_graph_layer(frame)
    elif activeLaneMap in (1, 2):
        frame = _render_mask_layer(frame, activeLaneMap)

    # --- minimap in top-right corner of the drawing area (unchanged) ---
    crop = cv2.resize(
        frame,
        (
            int(float(VIEW_W // zoom) / imageDim[0] * 256),
            int(float(VIEW_H // zoom) / imageDim[1] * 256),
        ),
        interpolation=cv2.INTER_LANCZOS4,
    )
    r1 = int(float(pos[1]) / imageDim[0] * 256)
    c1 = int(float(pos[0]) / imageDim[1] * 256)
    r2 = r1 + np.shape(crop)[0]
    c2 = c1 + np.shape(crop)[1]
    minimap[r1:r2, c1:c2, :] = crop

    if not vis_switch_no_minimap:
        frame[10:10+256, VIEW_W-256-10:VIEW_W-10, :] = minimap
        cv2.rectangle(frame, (VIEW_W-256-10, 10+256), (VIEW_W-10, 10), (255, 255, 255), 2)
        x1 = VIEW_W-256-10 + int(float(pos[0]) / imageDim[0] * 256)
        x2 = VIEW_W-256-10 + int(float(pos[0] + VIEW_W // zoom) / imageDim[0] * 256)
        y1 = 10 + int(float(pos[1] + VIEW_H // zoom) / imageDim[0] * 256)
        y2 = 10 + int(float(pos[1]) / imageDim[0] * 256)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # crosshair / erase box
    color = (255, 255, 255)
    if editingMode == "autofill_stage1": color = (255, 0, 0)
    if editingMode == "ready_to_edit": color = (0, 255, 255)
    if deleteMode == "(delete)": color = (0, 0, 255)
    if editingMode == "erase":
        cv2.rectangle(frame,
                      (max(0, mousex - erase_size * 50), max(0, mousey - erase_size * 50)),
                      (min(VIEW_W - 1, mousex + erase_size * 50), min(VIEW_H - 1, mousey + erase_size * 50)),
                      (0, 0, 255), 2)
    else:
        cv2.line(frame, (max(0, mousex - 50), mousey), (min(VIEW_W - 1, mousex + 50), mousey), color, 1)
        cv2.line(frame, (mousex, max(0, mousey - 50)), (mousex, min(VIEW_H - 1, mousey + 50)), color, 1)

    # status line
    if not noshow:
        cv2.putText(frame, "%s | %s | Layer %d | Render %.3fs"
                    % (editingMode + deleteMode, edgeType, activeLaneMap, renderTime),
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # compose sidebar
    full = draw_sidebar(frame)

    renderTime = time() - t0
    if not noshow:
        cv2.imshow("image", full)
    return full

# -------------- Optional config image export (kept) --------------
if len(sys.argv) > 3 and os.path.isfile(sys.argv[3]):
    config = json.load(open(sys.argv[3]))
    # load the ONLY (or first) image for offline render
    load_image_and_init(image_paths[0])
    # visual switches
    vis_switch_no_direction = config.get("no_direction", vis_switch_no_direction)
    vis_switch_no_arrow = config.get("no_arrow", vis_switch_no_arrow)
    vis_switch_no_minimap = config.get("no_minimap", vis_switch_no_minimap)
    vis_switch_no_diff_way_and_link = config.get("no_diff_way_and_link", vis_switch_no_diff_way_and_link)
    vis_switch_no_vertices = config.get("no_vertices", vis_switch_no_vertices)
    vis_switch_no_way = config.get("no_way", vis_switch_no_way)
    vis_switch_no_link = config.get("no_link", vis_switch_no_link)
    if config.get("bk") == "white":
        image[:] = 255
    transpose = "transpose" in config
    frame = redraw(noshow=True, transpose=transpose)
    crop = config["crop"]
    img = frame[crop[1]:crop[3], crop[0]:crop[2]]
    if transpose:
        img = np.transpose(img, axes=[1, 0, 2])
    cv2.imwrite(config["output"], img)
    sys.exit(0)

# -------------- normal interactive mode --------------
# Load first image + its annotations
load_image_and_init(image_paths[img_idx])
annotation = current_annotation_path()
load_annotations_for_current()

cv2.namedWindow("image")
cv2.moveWindow("image", 0, 0)
cv2.setMouseCallback("image", mouseEventHandler)

def goto_image(new_idx: int):
    """Save current, switch to new image, load its annos; reset transient state."""
    global img_idx, laneMaps, laneMap, activeLaneMap, editingMode, deleteMode
    global lastNodeID, activeLinks, zoom, erase_size, pos
    save_annotations_for_current()
    img_idx = new_idx
    load_image_and_init(image_paths[img_idx])
    load_annotations_for_current()
    # reset transient
    editingMode = "ready_to_draw"
    deleteMode = " "
    lastNodeID = None
    activeLinks = []
    zoom = 1
    erase_size = 1
    pos = [0, 0]

while True:
    hasUpdate = False
    if step == 0:
        hasUpdate = True
    step += 1

    k = cv2.waitKey(60) & 0xFF
    if k == 27:  # ESC
        save_annotations_for_current()
        break

    # --- navigation within image ---
    if k == ord("w"):
        pos[1] = max(0, pos[1] - 100); hasUpdate = True
    if k == ord("a"):
        pos[0] = max(0, pos[0] - 100); hasUpdate = True
    if k == ord("s"):
        pos[1] = min(dim[0] - VIEW_H // zoom, pos[1] + 100); hasUpdate = True
    if k == ord("d"):
        pos[0] = min(dim[1] - VIEW_W // zoom, pos[0] + 100); hasUpdate = True

    # --- edit/draw toggles ---
    if k == ord("e"):
        editingMode = "ready_to_draw" if editingMode == "ready_to_edit" else "ready_to_edit"
        hasUpdate = True
    if k == ord("f"):
        editingMode = "autofill_stage1" if editingMode == "ready_to_draw" else "ready_to_draw"; hasUpdate = True
    if k == ord("q"):
        edgeType = "way" if edgeType == "link" else "link"; hasUpdate = True
    if k == ord("z"):  # undo (note: not working for deletions in your original comment)
        laneMap.undo(); hasUpdate = True
        save_annotations_for_current()
    if k == ord("c"):
        activeLinks = []; editingMode = "selecting_link"; hasUpdate = True
    if k == ord("1"):
        zoom = 1; hasUpdate = True
    if k == ord("2"):
        zoom = 2; hasUpdate = True
    if k == ord("3"):
        zoom = 3; hasUpdate = True
    if k == ord("m"):
        activeLaneMap = (activeLaneMap + 1) % 3   # cycle 0,1,2
        laneMap = laneMaps[activeLaneMap]
        edgeType = "way"
        hasUpdate = True
    if k == ord("x"):
        deleteMode = "(delete)" if deleteMode == " " else " "; hasUpdate = True
    if k == ord("r"):
        editingMode = "ready_to_draw" if editingMode == "erase" else "erase"; hasUpdate = True
    if k == ord("4"):
        erase_size = 1; hasUpdate = True
    if k == ord("5"):
        erase_size = 2; hasUpdate = True
    if k == ord("6"):
        erase_size = 4; hasUpdate = True

    # --- NEW: dataset next/prev ---
    if k == ord("n") and len(image_paths) > 1:
        new_idx = (img_idx + 1) % len(image_paths)
        goto_image(new_idx); hasUpdate = True
    if k == ord("p") and len(image_paths) > 1:
        new_idx = (img_idx - 1) % len(image_paths)
        goto_image(new_idx); hasUpdate = True
    if k == ord("t"):  # NEW: toggle minimap
        vis_switch_no_minimap = not vis_switch_no_minimap
        hasUpdate = True
    # mouse moved?
    if lastMousex != mousex or lastMousey != mousey:
        lastMousex, lastMousey = mousex, mousey
        hasUpdate = True

    if hasUpdate:
        redraw()

cv2.destroyAllWindows()
