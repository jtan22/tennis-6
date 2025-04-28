from dataclasses import dataclass
from typing import List, Tuple

from tennis.bounding_box import BoundingBox
from tennis.utils import get_player_height

@dataclass
class ReferencePlayer:
    """Representation of a reference player with convenience methods."""
    player_id: int
    original_bounding_box: BoundingBox
    reference_coordinate: Tuple[int, int]
    action: int
    net_clearance: float
    
    @property
    def height_meters(self) -> float:
        """Calculate the height of the player in meters."""
        return get_player_height(self.original_bounding_box.width_height_ratio)
