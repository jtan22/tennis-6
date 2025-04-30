from dataclasses import dataclass
from typing import Tuple

from tennis.bounding_box import BoundingBox

@dataclass
class ReferencePlayer:
    """Representation of a reference player with convenience methods."""
    player_id: int
    original_bounding_box: BoundingBox
    reference_coordinate: Tuple[int, int]
    action: int
    net_clearance: float
