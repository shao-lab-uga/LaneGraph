import json
from typing import Any, Dict, List, Union, Tuple
import networkx as nx
ENABLE_LOG = False


def dist2(p1, p2):
    a = p1[0] - p2[0]
    b = p1[1] - p2[1]
    return a * a + b * b


class LaneMap:
    def __init__(self):
        self.nodes: Dict[int, List[float]] = {}
        self.nid: int = 0

        # neighbors: directed adjacency (subset of neighbors_all)
        self.neighbors: Dict[int, List[int]] = {}
        # neighbors_all: undirected adjacency (symmetric)
        self.neighbors_all: Dict[int, List[int]] = {}

        # edgeType stored for ANY directed pair (u, v)
        self.edgeType: Dict[Tuple[int, int], str] = {}  # "way" or "link"
        self.nodeType: Dict[int, str] = {}

        self.history: List[Any] = []

    # ---------- existing methods ----------
    def updateNodeType(self):
        for nid, _pos in self.nodes.items():
            nei = self.neighbors_all.get(nid, [])
            if len(nei) == 0:
                self.nodeType[nid] = "way"
            else:
                allLink = True
                for nn in nei:
                    edge = (nid, nn)
                    if self.edgeType.get(edge, "way") == "way":
                        allLink = False
                        break
                self.nodeType[nid] = "link" if allLink else "way"

    def query(self, p, nodelist=None):
        if nodelist is None or len(nodelist) == 0:
            bestd, bestnid = 10 * 10, None
            for nid, pos in self.nodes.items():
                if self.nodeType.get(nid, "way") == "link":
                    continue
                d = dist2(p, pos)
                if d < bestd:
                    bestd = d
                    bestnid = nid
            return bestnid
        else:
            bestd, bestnid = 10 * 10, None
            for nid in nodelist:
                pos = self.nodes[nid]
                d = dist2(p, pos)
                if d < bestd:
                    bestd = d
                    bestnid = nid
            return bestnid

    def findLink(self, nid):
        nodelist = []
        queue = [nid]
        badnodes = []
        while len(queue) > 0:
            cur = queue.pop()
            for nn in self.neighbors_all.get(cur, []):
                if nn not in self.nodeType:
                    print("Error cannot find node %d in nodeType" % nn)
                    if nn not in badnodes:
                        badnodes.append(nn)
                else:
                    if self.nodeType[nn] == "link":
                        if nn not in nodelist:
                            nodelist.append(nn)
                            queue.append(nn)
        for n in badnodes:
            self.deleteNode(n)
        return nodelist

    def findAllPolygons(self):
        visited = set()
        polygons = []
        for nid, _ in self.nodes.items():
            if nid in visited:
                continue
            start = nid
            cur = nid
            polygon = []
            while True:
                polygon.append(cur)
                nei = self.neighbors.get(cur, [])
                if len(nei) != 1:
                    break
                cur = nei[0]
                if cur == start:
                    polygon.append(start)
                    break
                if cur in polygon:
                    break
            if len(polygon) > 1 and polygon[0] == polygon[-1]:
                polygons.append(polygon)
        return polygons

    def addNode(self, p, updateNodeType=True):
        self.history.append(["addNode", p, self.nid])
        self.nodes[self.nid] = list(p)
        self.neighbors[self.nid] = []
        self.neighbors_all[self.nid] = []
        self.nid += 1
        if updateNodeType:
            self.updateNodeType()
        return self.nid - 1

    def addEdge(self, n1, n2, edgetype="way", updateNodeType=True):
        self.history.append(["addEdge", n1, n2, edgetype])

        if n1 not in self.neighbors:
            self.neighbors[n1] = []
        if n1 not in self.neighbors_all:
            self.neighbors_all[n1] = []
        if n2 not in self.neighbors_all:
            self.neighbors_all[n2] = []

        if n2 not in self.neighbors[n1]:
            self.neighbors[n1].append(n2)

        if n2 not in self.neighbors_all[n1]:
            self.neighbors_all[n1].append(n2)
        if n1 not in self.neighbors_all[n2]:
            self.neighbors_all[n2].append(n1)

        edge = (n1, n2)
        self.edgeType[edge] = edgetype
        edge = (n2, n1)
        self.edgeType[edge] = edgetype
        if updateNodeType:
            self.updateNodeType()

    def deleteNode(self, nid):
        self.history.append(["deleteNode", nid])
        if ENABLE_LOG:
            print("delete", nid)

        if nid in self.neighbors_all:
            neilist = list(self.neighbors_all[nid])
            for nn in neilist:
                self.deleteEdge(nn, nid)
                self.deleteEdge(nid, nn)

        if nid in self.nodes:
            del self.nodes[nid]

        if nid in self.neighbors:
            del self.neighbors[nid]

        if nid in self.neighbors_all:
            del self.neighbors_all[nid]

        if nid in self.nodeType:
            del self.nodeType[nid]

    def deleteEdge(self, n1, n2):
        self.history.append(["deleteEdge", n1, n2])
        if ENABLE_LOG:
            print("delete edge", n1, n2)
        if n1 in self.neighbors and n2 in self.neighbors[n1]:
            self.neighbors[n1].remove(n2)
            if ENABLE_LOG:
                print(self.neighbors[n1], n2)
        if n1 in self.neighbors_all and n2 in self.neighbors_all[n1]:
            self.neighbors_all[n1].remove(n2)
            if ENABLE_LOG:
                print(self.neighbors_all[n1], n2)
        if n2 in self.neighbors_all and n1 in self.neighbors_all[n2]:
            self.neighbors_all[n2].remove(n1)
            if ENABLE_LOG:
                print(self.neighbors_all[n2], n1)

        if (n1, n2) in self.edgeType:
            del self.edgeType[(n1, n2)]

    def checkConsistency(self):
        for nid in self.nodes.keys():
            if nid not in self.neighbors_all:
                self.neighbors_all[nid] = []
                print("missing neighbors_all", nid)

        for nid in self.nodes.keys():
            for nn in list(self.neighbors_all.get(nid, [])):
                if nn not in self.neighbors_all:
                    print("unsolved error", nn)
                    continue
                if nid not in self.neighbors_all[nn]:
                    if ENABLE_LOG:
                        print("incomplete neighbors", nid, nn)
                    self.neighbors_all[nn].append(nid)

    def undo(self):
        if len(self.history) > 0:
            item = self.history.pop()
            if item[0] == "addNode":
                nid = item[2]
                if nid in self.nodes:
                    del self.nodes[nid]
                if nid in self.neighbors:
                    del self.neighbors[nid]
                if nid in self.neighbors_all:
                    del self.neighbors_all[nid]
                if nid in self.nodeType:
                    del self.nodeType[nid]

            elif item[0] == "addEdge":
                n1, n2, _edgeType = item[1], item[2], item[3]
                if n1 in self.neighbors and n2 in self.neighbors[n1]:
                    self.neighbors[n1].remove(n2)
                if n1 in self.neighbors_all and n2 in self.neighbors_all[n1]:
                    self.neighbors_all[n1].remove(n2)
                if n2 in self.neighbors_all and n1 in self.neighbors_all[n2]:
                    self.neighbors_all[n2].remove(n1)
                if (n1, n2) in self.edgeType:
                    del self.edgeType[(n1, n2)]

            self.updateNodeType()

    # ---------- JSON (de)serialization that PRESERVES DIRECTION ----------
    def to_dict(self, include_history: bool = False) -> Dict[str, Any]:
        """
        Build a fully JSON-serializable dictionary representing the graph.
        Directed edges are serialized exactly as stored.
        """
        return {
            "version": 2,
            "nid": int(self.nid),
            "nodes": {str(k): list(v) for k, v in self.nodes.items()},          # IDs as strings
            "neighbors": {str(k): list(v) for k, v in self.neighbors.items()},
            "neighbors_all": {str(k): list(v) for k, v in self.neighbors_all.items()},
            "edges": [
                {"u": int(u), "v": int(v), "type": etype}
                for (u, v), etype in self.edgeType.items()
            ],  # keep all directed entries
            "nodeType": {str(k): v for k, v in self.nodeType.items()},
            **({"history": self.history} if include_history else {}),
        }

    def to_json(self, path: str = None, indent: int = 2, include_history: bool = False) -> str:
        """
        Serialize to JSON string. If path is provided, also write to disk.
        """
        payload = self.to_dict(include_history=include_history)
        s = json.dumps(payload, indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        return s

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LaneMap":
        """
        Reconstruct a LaneMap from a dictionary created by to_dict().
        """
        lm = cls()

        # nodes
        lm.nodes = {int(k): list(v) for k, v in data.get("nodes", {}).items()}

        # neighbors / neighbors_all
        lm.neighbors = {int(k): list(v) for k, v in data.get("neighbors", {}).items()}
        lm.neighbors_all = {int(k): list(v) for k, v in data.get("neighbors_all", {}).items()}

        # nid (next id)
        lm.nid = int(data.get("nid", (max(lm.nodes.keys()) + 1) if lm.nodes else 0))

        # directed edges exactly as stored
        lm.edgeType = {}
        for e in data.get("edges", []):
            u, v = int(e["u"]), int(e["v"])
            etype = str(e.get("type", "way"))
            lm.edgeType[(u, v)] = etype

        # optional nodeType (otherwise recompute)
        node_type_in = data.get("nodeType")
        if isinstance(node_type_in, dict) and len(node_type_in) > 0:
            lm.nodeType = {int(k): str(v) for k, v in node_type_in.items()}
        else:
            lm.updateNodeType()

        # optional history
        if "history" in data:
            lm.history = list(data["history"])

        # sanity: ensure neighbor symmetry for neighbors_all
        lm.checkConsistency()
        # ensure nodeType is consistent with edges
        lm.updateNodeType()
        return lm

    @classmethod
    def from_json(cls, src: Union[str, Dict[str, Any]]) -> "LaneMap":
        """
        Load from a JSON file path or a raw JSON string or a dict.
        """
        if isinstance(src, dict):
            data = src
        else:
            # try path first
            try:
                with open(src, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (FileNotFoundError, OSError):
                # raw JSON string
                data = json.loads(src)
        return cls.from_dict(data)
