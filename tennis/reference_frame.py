from dataclasses import dataclass
from typing import List, Tuple

from tennis.reference_ball import ReferenceBall
from tennis.reference_player import ReferencePlayer

@dataclass
class ReferenceFrame:
    """Representation of a reference frame with convenience methods."""
    frame_number: int
    near_player_id: int
    far_player_id: int
    player_1: ReferencePlayer
    player_2: ReferencePlayer
    ball: ReferenceBall

    def get_near_player(self) -> ReferencePlayer:
        """Get the near player."""
        return self.player_1 if self.player_1.player_id == self.near_player_id else self.player_2

    def get_far_player(self) -> ReferencePlayer:
        """Get the far player."""
        return self.player_1 if self.player_1.player_id == self.far_player_id else self.player_2

