"""Canonical public and internal names for the GSPAR planning routes."""

from __future__ import annotations

PROJECT_NAME = "GSPAR"
PROJECT_FULL_NAME = "Gaussian Splatting Path Planning for Active Reconstruction"

GSPAR_DIRECT = "gspar_direct"

GSPAR_GLOBAL_ASTAR_EGO = "gspar_global_astar_ego"
NVBLOX_ESDF_ASTAR_STOMP = "nvblox_esdf_astar_stomp"

GSPAR_GLOBAL_ASTAR_EGO_LABEL = "GSPAR-Global-AStar-EGO"
GSPAR_DIRECT_LABEL = "GSPAR-Direct"
NVBLOX_ESDF_ASTAR_STOMP_LABEL = "nvblox-ESDF-AStar-STOMP"

METHOD_LABEL_BY_ID = {
    GSPAR_GLOBAL_ASTAR_EGO: GSPAR_GLOBAL_ASTAR_EGO_LABEL,
    GSPAR_DIRECT: GSPAR_DIRECT_LABEL,
    NVBLOX_ESDF_ASTAR_STOMP: NVBLOX_ESDF_ASTAR_STOMP_LABEL,
}

def canonical_method_label(value: str) -> str:
    """Return the formal label while preserving unknown third-party methods."""

    method = str(value)
    return METHOD_LABEL_BY_ID.get(method, method)
