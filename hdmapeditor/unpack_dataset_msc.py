import json
import pickle
import sys

import numpy as np
import os
import math

from roadstructure import LaneMap  # noqa Needed for pickle
import cv2
import pathlib

sys.path.append(os.path.dirname(sys.path[0]))  # Needed for pickle
input_folder = sys.argv[1]
output_folder = sys.argv[2]

pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

def find_centers(node_links, node_coordinates, threshold):

    def calculate_centroid(node_ids, coordinates_dict, int_flag=False):
        if not node_ids:  # If no input
            return 0, 0

        x_total = 0
        y_total = 0
        count = len(node_ids)

        for node_id in node_ids:
            x_coord, y_coord = coordinates_dict[node_id]  # Get coords from dictionary
            x_total += x_coord
            y_total += y_coord
        if int_flag:
            return int(np.round(x_total / count)), int(np.round(y_total / count))
        return x_total / count, y_total / count  # Return the centroid's x and y

    def calculate_distance(center1, center2):
        return math.sqrt(
            (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
        )

    def merge_sublists(lists):
        merged = []

        while lists:
            current = lists.pop(0)  # get first sublist
            merged_current = set(current)  # set for ovelaps

            merged_flag = True  # overlap flag

            while merged_flag:
                merged_flag = False
                for sublist in lists[:]:  # shallow copy of lists
                    if merged_current.intersection(sublist):
                        merged_current.update(sublist)
                        lists.remove(sublist)
                        merged_flag = True

            merged.append(list(merged_current))

        return merged

    def merge_sublists_by_coordinates(lists, coordinates_dict):
        merged = []
        while lists:
            current = lists.pop(0)  # Take the first sublist
            merged_current = set(current)  # Use a set to eliminate duplicates

            # Calculate the centroid for the current sublist
            current_centroid = calculate_centroid(current, coordinates_dict)

            merged_flag = True

            while merged_flag:
                merged_flag = False
                for sublist in lists[
                    :
                ]:  # Iterate over a copy to avoid modification during iteration
                    # Calculate the centroid for the next sublist
                    sublist_centroid = calculate_centroid(sublist, coordinates_dict)

                    # Check if the distance between centroids is less than 200
                    if (
                        calculate_distance(current_centroid, sublist_centroid)
                        < threshold
                    ):
                        # Merge the lists
                        merged_current.update(sublist)
                        lists.remove(sublist)  # Remove the merged sublist
                        merged_flag = True  # Set the flag to check again

                        # Update the centroid since the list has now been merged
                        current_centroid = calculate_centroid(
                            merged_current, coordinates_dict
                        )

            merged.append(list(merged_current))  # Add the merged set to the result

        return merged

    node_links = [[key] + value for key, value in node_links.items()]
    node_links = merge_sublists(node_links)

    valid_node_ids = set(node_coordinates.keys())
    node_links = [[node for node in sublist if node in valid_node_ids] for sublist in node_links]

    merged_graphs = merge_sublists_by_coordinates(node_links, node_coordinates)
    targets = []
    for sublist in merged_graphs:
        center = calculate_centroid(sublist, node_coordinates, int_flag=True)
        targets.append(center)
    return targets


counter = 0  # counter for image number
total_length = 0  # Length of lines in pixels ?

with open(input_folder + "/regions.json") as file:
    regions = json.load(file)

print(len(f"{regions} regions"))

for region in regions:

    img = cv2.imread(input_folder + f"/sat_{counter}.jpg")

    try:
        with open(input_folder + f"/sat_{counter}_label.p", "rb") as file:
            labels = pickle.load(file)
    except Exception as e:
        print(e)
        break
    roadlabel, masklabel = labels

    adv = 0
    for nid, p1 in roadlabel.nodes.items():
        for nn in roadlabel.neighbors[nid]:
            p2 = roadlabel.nodes[nn]
            L = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            adv += L
    total_length += adv

    polygons = masklabel.findAllPolygons()

    # Draw masks, normal, and lanes
    sat = img
    mask = np.zeros_like(sat) + 255
    normal = np.zeros_like(sat) + 127
    lane = np.zeros_like(sat)
    margin = 128

    # draw mask
    for polygon in polygons:
        polygon_list = []
        x2 = 0
        y2 = 0
        for i in range(len(polygon) - 1):
            x1 = masklabel.nodes[polygon[i]][0] - margin
            y1 = masklabel.nodes[polygon[i]][1] - margin
            x2 = masklabel.nodes[polygon[i + 1]][0] - margin
            y2 = masklabel.nodes[polygon[i + 1]][1] - margin

            polygon_list.append([x1, y1])
        polygon_list.append([x2, y2])

        area = np.array(polygon_list)
        area = area.astype(np.int32)
        cv2.fillPoly(mask, [area], (0, 0, 0))

    # draw lanes and direction
    for nid, nei in roadlabel.neighbors.items():
        x1 = roadlabel.nodes[nid][0] - margin
        y1 = roadlabel.nodes[nid][1] - margin

        for nn in nei:
            if roadlabel.edgeType[(nid, nn)] != "way":
                continue

            x2 = roadlabel.nodes[nn][0] - margin
            y2 = roadlabel.nodes[nn][1] - margin

            dx = x2 - x1
            dy = y2 - y1
            l = math.sqrt(float(dx * dx + dy * dy)) + 0.001
            dx /= l
            dy /= l

            color = (127 + int(dx * 127), 127 + int(dy * 127), 127)

            cv2.line(lane, (x1, y1), (x2, y2), (255, 255, 255), 5)
            cv2.line(normal, (x1, y1), (x2, y2), color, 5)

    cv2.imwrite(output_folder + f"/sat_{counter}.jpg", sat)
    cv2.imwrite(output_folder + f"/regionmask_{counter}.jpg", mask)
    cv2.imwrite(output_folder + f"/lane_{counter}.jpg", lane)
    cv2.imwrite(output_folder + f"/normal_{counter}.jpg", normal)
    print(f"Saved image set {counter}")

    # Create link files
    terminal_nodes = []
    for nid in roadlabel.nodes.keys():
        way_c = 0
        link_c = 0
        for nn in roadlabel.neighbors_all[nid]:
            if (nid, nn) in roadlabel.edgeType:
                edgetype = roadlabel.edgeType[(nid, nn)]
            else:
                edgetype = roadlabel.edgeType[(nn, nid)]

            if edgetype == "way":
                way_c += 1
            else:
                link_c += 1

        if way_c > 0 and link_c > 0:
            terminal_nodes.append(nid)

    linkset = set()
    links = []
    for nid in terminal_nodes:
        link = [nid]
        queue = [[nid, [nid]]]
        while len(queue) > 0:
            cur, curlist = queue.pop()
            for nn in roadlabel.neighbors[cur]:
                if roadlabel.edgeType[(cur, nn)] == "way":
                    continue
                if roadlabel.nodeType[nn] == "link":
                    newlist = list(curlist)
                    newlist.append(nn)
                    queue.append([nn, newlist])
                else:
                    newlist = list(curlist)
                    newlist.append(nn)
                    if (nid, nn) not in linkset:
                        linkset.add((nid, nn))
                    if len(newlist) > 1:
                        links.append(list(newlist))
    nidmap = {}

    for item in linkset:
        n1, n2 = item
        if n1 not in nidmap:
            nidmap[n1] = [n2]
        else:
            nidmap[n1].append(n2)

        if n2 not in nidmap:
            nidmap[n2] = [n1]
        else:
            nidmap[n2].append(n1)

    for nid_check in terminal_nodes:
        if nid_check not in nidmap:
            nidmap[nid_check] = []

    locallinks = []
    for link in links:
        vertices = []
        outOfRange = False
        for nid_link in link:
            x = roadlabel.nodes[nid_link][0] - margin
            y = roadlabel.nodes[nid_link][1] - margin

            if 0 < x < 4096 and 0 < y < 4096 and mask[y, x, 0] > 127:
                vertices.append([x, y])
            else:
                outOfRange = True
        if not outOfRange:
            locallinks.append(vertices)

    localnodes = {}
    for nid in terminal_nodes:
        x = roadlabel.nodes[nid][0] - margin
        y = roadlabel.nodes[nid][1] - margin
        if 0 < x < 4096 and 0 < y < 4096 and mask[y, x, 0] > 127:
            localnodes[nid] = [x, y]

    with open(output_folder + f"/link_{counter}.json", "w") as file:
        json.dump([nidmap, localnodes, locallinks, find_centers(nidmap, localnodes, 200)], file, indent=2)  # type: ignore

    counter += 1
