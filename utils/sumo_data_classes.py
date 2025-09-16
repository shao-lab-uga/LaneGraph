import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Tuple

class SUMONode:
    def __init__(self, id: Optional[str] = None, x: float = 0.0, y: float = 0.0, node_type: str = "priority") -> None:
        self.id: Optional[str] = id
        self.x: float = x
        self.y: float = y
        self.node_type: str = node_type 

    def init_node_from_xml(self, xml_element: ET.Element) -> None:
        self.id = xml_element.get("id")
        self.x = float(xml_element.get("x", "0"))
        self.y = float(xml_element.get("y", "0"))
        self.node_type = xml_element.get("type", "")  # keep original attribute name "type" in XML

    def get_node_as_xml(self) -> ET.Element:
        node = ET.Element("node")
        if self.id is None:
            raise ValueError("SUMONode.id is required before serialization")
        node.set("id", self.id)
        node.set("x", str(self.x))
        node.set("y", str(self.y))
        if self.node_type:
            node.set("type", self.node_type)
        return node

    def __str__(self) -> str:
        return f"SUMONode(id={self.id}, x={self.x}, y={self.y}, type={self.node_type})"

class SUMOLane:
    def __init__(self, index: int = 0, speed: float = 27.78, lane_type: str = "driving") -> None:
        self.index: int = index
        self.speed: float = speed
        self.lane_type: str = lane_type

    def init_lane_from_xml(self, xml_element: ET.Element) -> None:
        # SUMO lane index in plain .edg.xml is an integer
        self.index = int(xml_element.get("index", "0"))
        # speed may be absent; default to 0.0
        self.speed = float(xml_element.get("speed", "0"))
        self.lane_type = xml_element.get("type", "")

    def get_lane_as_xml(self) -> ET.Element:
        lane = ET.Element("lane")
        lane.set("index", str(self.index))
        lane.set("speed", f"{self.speed:g}")
        if self.lane_type:
            lane.set("type", self.lane_type)
        return lane


class SUMOEdge:
    def __init__(
        self,
        id: str = "",
        from_node: str = "",
        to_node: str = "",
        edge_type: str = "",
        numLanes: int = 0,
        shape: str = "",
        disallow: Optional[str] = None,
        allow: Optional[str] = None
    ) -> None:
        self.id: str = id
        self.from_node: str = from_node
        self.to_node: str = to_node
        self.edge_type: str = edge_type
        self.numLanes: int = numLanes
        self.shape: str = shape
        self.disallow: Optional[str] = disallow
        self.allow: Optional[str] = allow
        self.lanes: Dict[str, SUMOLane] = {}

    def init_edge_from_xml(self, xml_element: ET.Element) -> None:
        self.id = xml_element.get("id", "")
        self.from_node = xml_element.get("from", "")
        self.to_node = xml_element.get("to", "")
        self.edge_type = xml_element.get("type", "")
        # numLanes may be missing in plain files if per-lane elements are used
        num_lanes_attr = xml_element.get("numLanes")
        self.numLanes = int(num_lanes_attr) if num_lanes_attr is not None else 0
        self.shape = xml_element.get("shape", "")
        self.disallow = xml_element.get("disallow")
        self.allow = xml_element.get("allow")

    def get_edge_as_xml(self) -> ET.Element:
        edge = ET.Element("edge")
        edge.set("id", self.id)
        edge.set("from", self.from_node)
        edge.set("to", self.to_node)
        if self.edge_type:
            edge.set("type", self.edge_type)
        if self.numLanes:
            edge.set("numLanes", str(self.numLanes))
        if self.shape:
            edge.set("shape", self.shape)
        if self.disallow:
            edge.set("disallow", self.disallow)
        if self.allow:
            edge.set("allow", self.allow)
        # (Optional) serialize lanes if you want them inline in .edg.xml
        for lane_index, lane in self.lanes.items():
            edge.append(lane.get_lane_as_xml())
        return edge
    
    def shape_string_to_coordinates(self):
        if not self.shape:
            return []
        # Split the shape string into individual points
        points = self.shape.split(" ")
        coordinates = []
        for point in points:
            x, y = point.split(",")
            coordinates.append((float(x), float(y)))
        return coordinates

    def coordinates_to_shape_string(self, coordinates: List[Tuple[float, float]]) -> str:
        
        self.shape = " ".join(f"{x},{y}" for x, y in coordinates)

        return self.shape

    def __str__(self) -> str:
        return f"SUMOEdge(id={self.id}, from={self.from_node}, to={self.to_node}, type={self.edge_type}, numLanes={self.numLanes}, shape={self.shape}, disallow={self.disallow})"

class SUMOConnection:
    def __init__(self, 
                 from_edge: str = "", 
                 to_edge: str = "",
                 from_lane: str = "",
                 to_lane: str = "") -> None:
        self.from_edge = from_edge
        self.to_edge = to_edge
        self.from_lane = from_lane
        self.to_lane = to_lane

    def init_connection_from_xml(self, xml_element: ET.Element) -> None:
        self.from_edge = xml_element.get("from", "")
        self.to_edge = xml_element.get("to", "")
        self.from_lane = xml_element.get("fromLane", "")
        self.to_lane = xml_element.get("toLane", "")

    def get_edge_as_xml(self) -> ET.Element:
        conn = ET.Element("connection")
        conn.set("from", self.from_edge)
        conn.set("to", self.to_edge)
        conn.set("fromLane", self.from_lane)
        conn.set("toLane", self.to_lane)
        return conn

    def __str__(self) -> str:
        return f"SUMOConnection(from_edge={self.from_edge}, to_edge={self.to_edge}, from_lane={self.from_lane}, to_lane={self.to_lane})"