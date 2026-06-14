"""Road-graph neighbourhood builder.

Parses a SUMO `.net.xml` to build the intersection adjacency graph
(which traffic lights are connected by a road segment).  Used by the
spatial-coordination transformer (Phase 2) to define each agent's
neighbourhood.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Set


def build_adjacency(net_xml: str, tls_ids: List[str]) -> Dict[str, List[str]]:
    """Parse a SUMO `.net.xml` and return a 1-hop adjacency graph for TLS nodes.

    Args:
        net_xml: path to the SUMO `.net.xml` file.
        tls_ids: list of traffic-light junction IDs to include.

    Returns:
        dict mapping each TLS id to its sorted list of neighbouring TLS ids.
    """
    tree = ET.parse(net_xml)
    root = tree.getroot()
    tls_set = set(tls_ids)

    # 1.  lane -> junction mapping from <junction incLanes="...">.
    lane_to_junc: Dict[str, str] = {}
    for j in root.findall(".//junction"):
        jid = j.get("id")
        if jid in tls_set:
            for lane in (j.get("incLanes") or "").split():
                lane_to_junc[lane] = jid

    # 2.  edge -> junction mapping (strip the lane suffix).
    edge_to_junc: Dict[str, str] = {}
    for lane, junc in lane_to_junc.items():
        edge = lane.rsplit("_", 1)[0]          # "A1A0_0" -> "A1A0"
        edge_to_junc[edge] = junc

    # 3.  Build adjacency from <connection> elements.
    adj: Dict[str, Set[str]] = {tl: set() for tl in tls_ids}
    for conn in root.findall(".//connection"):
        from_edge = conn.get("from")
        to_edge = conn.get("to")
        if not from_edge or not to_edge:
            continue
        from_junc = edge_to_junc.get(from_edge)
        to_junc = edge_to_junc.get(to_edge)
        if from_junc and to_junc and from_junc != to_junc:
            if from_junc in tls_set and to_junc in tls_set:
                adj[from_junc].add(to_junc)

    return {k: sorted(v) for k, v in adj.items()}


def build_neighbour_tokens(
    agent_id: str,
    obs_dict: Dict[str, "np.ndarray"],
    adj: Dict[str, List[str]],
    max_neighbours: int = 8,
    include_self: bool = True,
) -> "np.ndarray":
    """Stack the agent's own obs + its neighbours' obs into a token matrix.

    Returns:
        ndarray of shape ``(1 + K, obs_dim)`` where K = min(neighbour_count, max_neighbours).
        If the agent has fewer neighbours than max_neighbours, the remaining rows are zero-padded.
    """
    import numpy as np

    self_obs = obs_dict[agent_id].flatten().astype(np.float32)
    obs_dim = self_obs.shape[0]
    neighbours = adj.get(agent_id, [])
    k = min(len(neighbours), max_neighbours)

    tokens = np.zeros((1 + k, obs_dim), dtype=np.float32)
    if include_self:
        tokens[0] = self_obs
    for i, n_id in enumerate(neighbours[:k]):
        tokens[1 + i] = obs_dict[n_id].flatten().astype(np.float32)

    return tokens
