import xml.etree.ElementTree as ET
from pyproj import CRS
from typing import Any, Callable, List, Sequence, Dict, TypeVar
from utils.sumo_data_classes import SUMONode, SUMOEdge, SUMOLane, SUMOConnection
T = TypeVar("T")

def list2string(lst: Sequence[Any]) -> str:
    """Convert a sequence to a space-separated string."""
    return " ".join(str(item) for item in lst)


def string2list(s: str, caster: Callable[[str], T]) -> List[T]:
    """
    Convert a space-separated string to a list using a caster, e.g. int/float/str.

    Examples:
        string2list("1 2 3", int)   -> [1, 2, 3]
        string2list("1.0 2.5", float) -> [1.0, 2.5]
        string2list("a b", str)     -> ["a", "b"]
    """
    # split() without args handles multiple spaces robustly
    return [caster(tok) for tok in s.split()]

def parse_SUMO_nodes(sumo_nodes_path: str) -> List[SUMONode]:
    tree = ET.parse(sumo_nodes_path)
    root = tree.getroot()
    loc = root.find("location")
    if loc is None:
        raise ValueError("No <location> tag in .nod.xml")
    net_offset = loc.get("netOffset")          # e.g. "-279962.57,-3756336.62"
    proj_param = loc.get("projParameter")      # e.g. "+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs"
    if net_offset is None or proj_param is None:
        raise ValueError("<location> must have netOffset and projParameter")

    net_offset = tuple(map(float, net_offset.split(",")))
    crs_proj = CRS.from_user_input(proj_param)
    nodes: Dict[str, SUMONode] = {}
    for elem in root.findall("node"):
        node = SUMONode()
        node.init_node_from_xml(elem)
        nodes[node.id] = node
    return nodes, crs_proj, net_offset


def parse_SUMO_edges(sumo_edges_path: str) -> List[SUMOEdge]:
    tree = ET.parse(sumo_edges_path)
    root = tree.getroot()
    loc = root.find("location")
    if loc is None:
        raise ValueError("No <location> tag in .nod.xml")
    net_offset = loc.get("netOffset")          # e.g. "-279962.57,-3756336.62"
    proj_param = loc.get("projParameter")      # e.g. "+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs"
    if net_offset is None or proj_param is None:
        raise ValueError("<location> must have netOffset and projParameter")

    net_offset = tuple(map(float, net_offset.split(",")))
    crs_proj = CRS.from_user_input(proj_param)
    print(f"Parsing SUMO edges from {sumo_edges_path} with offset {net_offset} and CRS {crs_proj.to_string()}")
    edges: Dict[str, SUMOEdge] = {}
    for elem in root.findall("edge"):
        edge = SUMOEdge()
        edge.init_edge_from_xml(elem)
        # gather lane children
        for lane_elem in elem.findall("lane"):
            lane = SUMOLane()
            lane.init_lane_from_xml(lane_elem)
            edge.lanes[lane.index] = lane
        # If lanes present but numLanes is 0/missing, set it from children
        if edge.numLanes == 0 and edge.lanes:
            edge.numLanes = len(edge.lanes)
        edges[edge.id] = edge
    return edges, net_offset, crs_proj

def parse_SUMO_connections(sumo_connections_path: str) -> List[SUMOConnection]:
    tree = ET.parse(sumo_connections_path)
    root = tree.getroot()
    connections_from_to_dict: Dict[str, SUMOConnection] = {}
    connections_to_from_dict: Dict[str, SUMOConnection] = {}

    for elem in root.findall("connection"):
        
        conn = SUMOConnection()
        conn.init_connection_from_xml(elem)
        connections_from_to_dict[conn.from_edge] = conn
        connections_to_from_dict[conn.to_edge] = conn
    return connections_from_to_dict, connections_to_from_dict