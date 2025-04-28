from dataclasses import dataclass
from typing import List, Tuple, Optional

from tennis.bounding_box import BoundingBox

@dataclass
class ReferenceBall:
    """Representation of a reference ball with convenience methods."""
    ball_id: int
    original_bounding_box: BoundingBox
    reference_coordinate: Tuple[int, int]
    height_meters: float
    horizontal_velocity: float
    vertical_velocity: float
    action: int
    
    @property
    def velocity(self) -> float:
        """Calculate the velocity of the ball."""
        return (self.horizontal_velocity**2 + self.vertical_velocity**2)**0.5
