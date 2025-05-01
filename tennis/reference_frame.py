from dataclasses import dataclass

from tennis.reference_ball import ReferenceBall
from tennis.reference_player import ReferencePlayer

@dataclass
class ReferenceFrame:
    """Representation of a reference frame with convenience methods."""
    frame_number: int
    near_player: ReferencePlayer
    far_player: ReferencePlayer
    ball: ReferenceBall

