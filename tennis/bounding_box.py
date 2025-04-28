from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class BoundingBox:
    """Representation of a bounding box with convenience methods."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        """Get width of bounding box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """Get height of bounding box."""
        return self.y2 - self.y1
    
    @property
    def width_height_ratio(self) -> float:
        """Get width to height ratio of bounding box."""
        if self.height == 0:
            return 0
        return self.width / self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def __repr__(self) -> str:
        """String representation of bounding box."""
        return f"[{round(self.x1)}, {round(self.y1)}, {round(self.x2)}, {round(self.y2)}]"
    
    def to_list(self) -> Tuple[int, int, int, int]:
        """Convert to list format."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    @classmethod
    def from_list(cls, coords: Tuple[int, int, int, int]) -> 'BoundingBox':
        """Create bounding box from list."""
        return cls(int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
    
    @classmethod
    def from_center_and_size(cls, center_x: float, center_y: float, 
                             width: float, height: float) -> 'BoundingBox':
        """Create bounding box from center point and dimensions."""
        half_width = width / 2
        half_height = height / 2
        return cls(
            int(center_x - half_width),
            int(center_y - half_height),
            int(center_x + half_width),
            int(center_y + half_height)
        )


