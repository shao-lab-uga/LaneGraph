
import json
import math

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import json
import cv2
import pickle
from roadstructure import LaneMap
import networkx as nx
import pickle
regions = json.load(open(sys.argv[1]))
inputfolder = sys.argv[2]
outputfolder = sys.argv[3]
os.mkdir(outputfolder) if not os.path.exists(outputfolder) else None
counter = 0
counter_out = 0
total_length = 0
for region in regions:
    min_lat, min_lon = region["lat"], region["lon"]
    region_tag = region["tag"]
    regionsize = 4096
    stride = 4096
    tilesize = 4096
    res = 8

    blocks = [region["ilat"], region["ilon"]]
    folder = inputfolder

    for ilat in range(blocks[0]):
        for ilon in range(blocks[1]):
            subregion = [
                min_lat + ilat * stride / res / 111111.0,
                min_lon
                + ilon * stride / res / 111111.0 / math.cos(math.radians(min_lat)),
            ]
            subregion += [
                min_lat + (ilat * stride + tilesize) / res / 111111.0,
                min_lon
                + (ilon * stride + tilesize)
                / res
                / 111111.0
                / math.cos(math.radians(min_lat)),
            ]

            img = cv2.imread(folder + "/sat_%d.jpg" % (counter))
            try:
                labels = pickle.load(open(folder + "/sat_%d_label.p" % (counter), "rb"))
            except:
                break
            roadlabel: LaneMap
            masklabel: LaneMap
            roadlabel, masklabel = labels

            adv = 0
            for nid, p1 in roadlabel.nodes.items():
                for nn in roadlabel.neighbors[nid]:
                    p2 = roadlabel.nodes[nn]
                    L = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                    total_length += L
                    adv += L

            print(adv / 8 / 1000.0)

            polygons = masklabel.findAllPolygons()
            # render masks, lanes, and normals (directions)


            sr = 0
            sc = 0
            sat = img[sr : sr + 4096, sc : sc + 4096, :]
            mask = np.zeros_like(sat) + 255
            normal = np.zeros_like(sat) + 127
            lane = np.zeros_like(sat)
            margin = 128
            # for ease of evaluations, the ground truth nx.Graph should be constructed
            # 1) undirected graph at non-intersection areas
            # 2) directed graph at non-intersection areas
            # 3) final full directed graph
            undirected_nonintersection_graph = nx.Graph()
            directed_nonintersection_graph = nx.DiGraph()
            full_directed_graph = nx.DiGraph()
            # add nodes first
            node_position_id_map = {}
            for nid, p in roadlabel.nodes.items():
                node_x = p[0] - sc - margin
                node_y = p[1] - sr - margin
                if node_x < 0 or node_x >= 4096 or node_y < 0 or node_y >= 4096:
                    continue
                node_type = roadlabel.nodeType[nid]
                
                if node_type == "way":
                    undirected_nonintersection_graph.add_node(nid, pos=(node_x, node_y), type='way')
                    directed_nonintersection_graph.add_node(nid, pos=(node_x, node_y), type='way')
                    full_directed_graph.add_node(nid, pos=(node_x, node_y), type='way')
                elif node_type == "link":
                    # the link nodes should be only added to the full directed graph
                    full_directed_graph.add_node(nid, pos=(node_x, node_y), type='link')
                else:
                    raise ValueError("Unknown node type: %s" % node_type)
                node_position_id_map[(node_x, node_y)] = nid
            # add edges
            for nid, nei in roadlabel.neighbors.items():
                node_x1 = roadlabel.nodes[nid][0] - sc - margin
                node_y1 = roadlabel.nodes[nid][1] - sr - margin
                for nn in nei:
                    node_x2 = roadlabel.nodes[nn][0] - sc - margin
                    node_y2 = roadlabel.nodes[nn][1] - sr - margin
                    # if edge out of range, ignore
                    if (node_x1, node_y1) in node_position_id_map and (node_x2, node_y2) in node_position_id_map:
                        if roadlabel.edgeType[(nid, nn)] == "way":
                            # add both direction for the undirected graph
                            undirected_nonintersection_graph.add_edge(nid, nn)
                            undirected_nonintersection_graph.add_edge(nn, nid)
                            # add only the directed edge for the directed graph
                            directed_nonintersection_graph.add_edge(nid, nn)
                            # add only the directed edge for the full directed graph
                            full_directed_graph.add_edge(nid, nn)
                        elif roadlabel.edgeType[(nid, nn)] == "link":
                            # only add the directed edge for the full directed graph
                            full_directed_graph.add_edge(nid, nn)
            # save the graphs
            with open(os.path.join(outputfolder, f"undirected_nonintersection_graph_{counter_out}.gpickle"), 'wb') as f:
                pickle.dump(undirected_nonintersection_graph, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(outputfolder, f"directed_nonintersection_graph_{counter_out}.gpickle"), 'wb') as f:
                pickle.dump(directed_nonintersection_graph, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(outputfolder, f"full_directed_graph_{counter_out}.gpickle"), 'wb') as f:
                pickle.dump(full_directed_graph, f, pickle.HIGHEST_PROTOCOL)
            print("number of nodes", undirected_nonintersection_graph.number_of_nodes(), directed_nonintersection_graph.number_of_nodes(), full_directed_graph.number_of_nodes())
            # draw mask
            for polygon in polygons:
                polygon_list = []
                for i in range(len(polygon) - 1):
                    x1 = masklabel.nodes[polygon[i]][0] - sc - margin
                    y1 = masklabel.nodes[polygon[i]][1] - sr - margin
                    x2 = masklabel.nodes[polygon[i + 1]][0] - sc - margin
                    y2 = masklabel.nodes[polygon[i + 1]][1] - sr - margin
                    polygon_list.append([x1, y1])
                polygon_list.append([x2, y2])
                area = np.array(polygon_list)
                area = area.astype(np.int32)
                cv2.fillPoly(mask, [area], (0, 0, 0))
            # draw lane and direction
            for nid, nei in roadlabel.neighbors.items():
                x1 = roadlabel.nodes[nid][0] - sc - margin
                y1 = roadlabel.nodes[nid][1] - sr - margin
                for nn in nei:
                    x2 = roadlabel.nodes[nn][0] - sc - margin
                    y2 = roadlabel.nodes[nn][1] - sr - margin
                    if roadlabel.edgeType[(nid, nn)] == "way":
                        dx = x2 - x1
                        dy = y2 - y1
                        l = math.sqrt(float(dx * dx + dy * dy)) + 0.001
                        dx /= l
                        dy /= l
                        color = (127 + int(dx * 127), 127 + int(dy * 127), 127)
                        cv2.line(lane, (x1, y1), (x2, y2), (255, 255, 255), 5)
                        cv2.line(normal, (x1, y1), (x2, y2), color, 5)
                        
            cv2.imwrite(outputfolder + "/sat_%d.jpg" % (counter_out), sat)
            cv2.imwrite(
                outputfolder + "/regionmask_%d.jpg" % (counter_out), mask
            )
            cv2.imwrite(outputfolder + "/lane_%d.jpg" % (counter_out), lane)
            cv2.imwrite(outputfolder + "/normal_%d.jpg" % (counter_out), normal)
            with open(
                outputfolder + "/region_%d.json" % (counter_out), "w"
            ) as f:
                json.dump(
                    {
                        "lat": subregion[0],
                        "lon": subregion[1],
                        "lat2": subregion[2],
                        "lon2": subregion[3],
                        "tag": region_tag,
                    },
                    f,
                )
            counter_out += 1
            print(counter_out)
            counter += 1

print(total_length, total_length / 8 / 1000.0)
