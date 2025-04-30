import cv2
import numpy as np
import copy
import pandas as pd
from scipy.spatial import distance
from typing import List, Dict, Tuple

from tennis.bounding_box import BoundingBox
from tennis.reference_frame import ReferenceFrame
from tennis.reference_player import ReferencePlayer
from tennis.reference_ball import ReferenceBall

from .utils import get_distance_between_points, \
    get_initial_horizontal_velocity, \
    get_horizontal_distance_by_time, \
    get_horizontal_velocity_by_time, \
    get_initial_vertical_velocity_hit, \
    get_initial_vertical_velocity_bounce, \
    get_vertical_distances_and_velocities, \
    get_hypotenuse
from .constants import BALL_FAR_BOUNCE, DOUBLES_LINE_WIDTH, DOUBLES_ALLEY_WIDTH, RUN_BACK_DEPTH, \
    SIDE_RUN_WIDTH, HALF_COURT_DEPTH, NO_MANS_LAND_DEPTH, REFERENCE_COURT_MARGIN_X, \
    REFERENCE_COURT_MARGIN_Y, CENTER_LINE_DEPTH, BALL_FAR_HIT, BALL_NEAR_BOUNCE, \
    BALL_NEAR_HIT, BALL_IN_FLIGHT, PLAYER_RUN, PLAYER_HIT

class ReferenceCourt:
    """
    The reference court is a scaled version of a standard tennis court.
    The court in our image is most likely an isosceles trapezium, but our 
    reference court is a rectangle.
    
    The only parameter we need to create our reference court is the width 
    of the court, we can work out the rest of the court dimensions.
    
    Here are the 14 key points of the reference court:
    
    0-4---------6-1
    | |         | |
    | |         | |
    | 8----C----9 |
    | |    |    | |
    | |    |    | |
    +-+----+----+-+
    | |    |    | |
    | |    |    | |
    | A----D----B |
    | |         | |
    | |         | |
    2-5---------7-3
    """
    
    def __init__(self, frame_width: int, frame_height: int, canvas_width: int):
        """
        Initialize the reference court with the given dimensions.
        
        Args:
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            canvas_width: Width of the canvas to draw the reference court
        """
        self.canvas_width = canvas_width
        self.pixel_to_meter_ratio = self.canvas_width / (DOUBLES_LINE_WIDTH + 2 * SIDE_RUN_WIDTH)
        print(f'pixel_to_meter_ratio: {self.pixel_to_meter_ratio}')
        self.canvas_depth = round(((HALF_COURT_DEPTH + RUN_BACK_DEPTH) * 2) * self.pixel_to_meter_ratio)

        # Initialize canvas positions on the frame
        self.canvas_x2 = frame_width - REFERENCE_COURT_MARGIN_X
        self.canvas_x1 = self.canvas_x2 - self.canvas_width
        self.canvas_y2 = frame_height - REFERENCE_COURT_MARGIN_Y
        self.canvas_y1 = self.canvas_y2 - self.canvas_depth

        print(f'canvas size: {self.canvas_width} x {round(self.canvas_depth)}')
        
        # Initialize court positions on the canvas
        self.court_x1 = SIDE_RUN_WIDTH * self.pixel_to_meter_ratio
        self.court_x2 = self.canvas_width - self.court_x1
        self.court_y1 = RUN_BACK_DEPTH * self.pixel_to_meter_ratio
        self.court_y2 = self.canvas_depth - self.court_y1

        print(f'court size: {round(self.court_x2 - self.court_x1)} x {round(self.court_y2 - self.court_y1)}')
        
        self.doubles_alley_width = DOUBLES_ALLEY_WIDTH * self.pixel_to_meter_ratio
        self.no_mans_land_depth = NO_MANS_LAND_DEPTH * self.pixel_to_meter_ratio
        
        # Initialize member variables that will be set later
        self.homography_matrix = None
        self.inverse_homography_matrix = None
        self.homographied_keypoints = []
        self.reference_frames = []
        
        self._initialize_keypoints()
        self._initialize_court_lines()
        self._initialize_homography_configurations()
    
    def _initialize_keypoints(self) -> None:
        """Initialize the 14 key points of the reference court."""
        self.keypoints = [
            (self.court_x1, self.court_y1),  # 0
            (self.court_x2, self.court_y1),  # 1
            (self.court_x1, self.court_y2),  # 2
            (self.court_x2, self.court_y2),  # 3
            (self.court_x1 + self.doubles_alley_width, self.court_y1),  # 4
            (self.court_x1 + self.doubles_alley_width, self.court_y2),  # 5
            (self.court_x2 - self.doubles_alley_width, self.court_y1),  # 6
            (self.court_x2 - self.doubles_alley_width, self.court_y2),  # 7
            (self.court_x1 + self.doubles_alley_width, self.court_y1 + self.no_mans_land_depth),  # 8
            (self.court_x2 - self.doubles_alley_width, self.court_y1 + self.no_mans_land_depth),  # 9
            (self.court_x1 + self.doubles_alley_width, self.court_y2 - self.no_mans_land_depth),  # 10
            (self.court_x2 - self.doubles_alley_width, self.court_y2 - self.no_mans_land_depth),  # 11
            ((self.court_x1 + self.court_x2) / 2, self.court_y1 + self.no_mans_land_depth),  # 12
            ((self.court_x1 + self.court_x2) / 2, self.court_y2 - self.no_mans_land_depth)   # 13
        ]
    
    def _initialize_court_lines(self) -> None:
        """Initialize the court lines connecting key points."""
        self.court_lines = [
            (0, 2),   # Left sideline
            (4, 5),   # Left singles sideline
            (6, 7),   # Right singles sideline
            (1, 3),   # Right sideline
            (0, 1),   # Top baseline
            (8, 9),   # Top service line
            (12, 13), # Center service line
            (10, 11), # Bottom service line
            (2, 3)    # Bottom baseline
        ]
    
    def _initialize_homography_configurations(self) -> None:
        """Initialize configurations for homography calculation."""
        self.homography_configurations = [
            [0, 1, 2, 3],    # doubles court
            [0, 6, 2, 7],    # right doubles alley plus singles court
            [4, 1, 5, 3],    # left doubles alley plus singles court
            [4, 6, 5, 7],    # singles court
            [0, 4, 2, 5],    # right doubles alley
            [6, 1, 7, 3],    # left doubles alley
            [4, 6, 10, 11],  # top no mans land plus mini court
            [10, 11, 5, 7]   # bottom no mans land
        ]

    def homograph_keypoints(self, given_keypoints: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Calculate homography matrix from reference keypoints to given keypoints.
        
        Args:
            given_keypoints: List of keypoints detected in the image
            
        Returns:
            List of homographied keypoints
        """
        self.homography_matrix = None
        distance_min = np.inf
        # Turn 2D keypoints into 3D, (14, 2) to (14, 1, 2)
        reference_keypoints = np.array(self.keypoints, dtype=np.float32).reshape((-1, 1, 2))
        
        # Try each homography configuration to find the best one
        for homography_configuration in self.homography_configurations:
            # The 4 reference keypoints
            reference_configuration = [self.keypoints[i] for i in homography_configuration]
            # The 4 given keypoints
            given_configuration = [given_keypoints[i] for i in homography_configuration]
            
            # Skip if any keypoint is missing
            if any(point is None for point in given_configuration):
                continue
                
            # Calculate homography matrix
            matrix, _ = cv2.findHomography(np.float32(reference_configuration), np.float32(given_configuration), method=cv2.RANSAC) # type: ignore
            
            # Apply homography to all reference keypoints
            transformed_keypoints = cv2.perspectiveTransform(reference_keypoints, matrix)
            
            # Calculate error for non-used keypoints
            distances = []
            for i in range(len(self.keypoints)):
                if i not in homography_configuration and given_keypoints[i] is not None:
                    distances.append(distance.euclidean(given_keypoints[i], transformed_keypoints[i][0]))
            
            # If no distances to compare, skip this configuration
            if not distances:
                continue
                
            distance_mean = np.mean(distances)
            if distance_mean < distance_min:
                self.homography_matrix = matrix
                distance_min = distance_mean
        
        if self.homography_matrix is None:
            raise ValueError("Could not find valid homography matrix with given keypoints")
            
        self.inverse_homography_matrix = np.linalg.inv(self.homography_matrix)
        
        # Apply homography to reference keypoints
        best_keypoints = cv2.perspectiveTransform(reference_keypoints, self.homography_matrix)
        # map(int, point) is to apply int function to each value in point
        self.homographied_keypoints = [(int(x), int(y)) for x, y in best_keypoints.reshape(-1, 2)]
        
        # Preserve detected keypoints where available
        for i in range(len(self.homographied_keypoints)):
            if given_keypoints[i] is not None:
                self.homographied_keypoints[i] = given_keypoints[i]
                
        return self.homographied_keypoints

    def get_reference_coordinate(self, original_coordinate: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert original coordinate to reference coordinate.
        
        Args:
            original_coordinate: Original coordinate in the image
            
        Returns:
            Reference coordinate
        """
        reshaped_original_coordinate = np.array(original_coordinate, dtype=np.float32).reshape((-1, 1, 2))
        reference_coordinates = cv2.perspectiveTransform(reshaped_original_coordinate, self.inverse_homography_matrix) # type: ignore
        reference_coordinate = reference_coordinates[0][0]
        return int(reference_coordinate[0]), int(reference_coordinate[1])

    def get_original_coordinate(self, reference_coordinate: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert reference coordinate to original coordinate.
        
        Args:
            reference_coordinate: Reference coordinate
            
        Returns:
            Original coordinate in the image
        """
        reshaped_reference_coordinate = np.array(reference_coordinate, dtype=np.float32).reshape((-1, 1, 2))
        original_coordinates = cv2.perspectiveTransform(reshaped_reference_coordinate, self.homography_matrix) # type: ignore
        original_coordinate = original_coordinates[0][0]
        return int(original_coordinate[0]), int(original_coordinate[1])

    def compute_reference_coordinates(self, 
                                player_positions: List[Dict[int, Tuple[int, int, int, int]]], 
                                near_player: int, 
                                far_player: int, 
                                ball_positions: List[Dict[int, Tuple[int, int, int, int]]], 
                                hits_and_bounces: List[int], 
                                fps: int) -> None:
        self.reference_frames = self._create_reference_frames(player_positions, near_player, far_player, ball_positions)
        self._compute_ball_coordinates(hits_and_bounces, fps)
        df = pd.DataFrame({
            'frame_number': [frame.frame_number for frame in self.reference_frames],
            'player_1_original_box': [frame.player_1.original_bounding_box for frame in self.reference_frames],
            'player_1_reference_point': [frame.player_1.reference_coordinate for frame in self.reference_frames],
            'player_1_action': [frame.player_1.action for frame in self.reference_frames],
            'player_1_net_clearance': [frame.player_1.net_clearance for frame in self.reference_frames],
            'player_2_original_box': [frame.player_2.original_bounding_box for frame in self.reference_frames],
            'player_2_reference_point': [frame.player_2.reference_coordinate for frame in self.reference_frames],
            'player_2_action': [frame.player_2.action for frame in self.reference_frames],
            'player_2_net_clearance': [frame.player_2.net_clearance for frame in self.reference_frames],
            'ball_original_box': [frame.ball.original_bounding_box for frame in self.reference_frames],
            'ball_reference_point': [frame.ball.reference_coordinate for frame in self.reference_frames],
            'ball_height': [frame.ball.height_meters for frame in self.reference_frames],
            'ball_horizontal_velocity': [frame.ball.horizontal_velocity for frame in self.reference_frames],
            'ball_vertial_velocity': [frame.ball.vertical_velocity for frame in self.reference_frames],
            'ball_action': [frame.ball.action for frame in self.reference_frames],
        })
        df.to_csv('reference_frames.csv', index=False)

    def _compute_ball_coordinates(self, hits_and_bounces: List[int], fps: int) -> None:
        i = 0
        # hits_and_bounces in the order of far_hit, near_bounce, near_hit, far_bounce
        while i < len(hits_and_bounces) - 1:
            # Skip if hit or bounce is missing
            if hits_and_bounces[i] is None or hits_and_bounces[i + 1] is None:
                i += 1
                continue
                
            print(f'Processing hit and bounce: {hits_and_bounces[i]} - {hits_and_bounces[i + 1]}')
            frame_count = hits_and_bounces[i + 1] - hits_and_bounces[i]

            hit_bounce_pattern = i % 4
            if hit_bounce_pattern == BALL_FAR_HIT:  # Far hit and bounce
                self._computer_far_hit_near_bounce(hits_and_bounces, i, frame_count, fps)
            elif hit_bounce_pattern == BALL_NEAR_BOUNCE:  # Near bounce and hit
                self._computer_near_bounce_near_hit(hits_and_bounces, i, frame_count, fps)
            elif hit_bounce_pattern == BALL_NEAR_HIT:  # Near hit and bounce
                self._computer_near_hit_far_bounce(hits_and_bounces, i, frame_count, fps)
            else:  # Far bounce and hit
                self._computer_far_bounce_far_hit(hits_and_bounces, i, frame_count, fps)

            i += 1

    def _create_reference_frames(self, 
                                player_positions: List[Dict[int, Tuple[int, int, int, int]]], 
                                near_player: int, 
                                far_player: int, 
                                ball_positions: List[Dict[int, Tuple[int, int, int, int]]]) -> List[ReferenceFrame]:
        reference_frames = []
        for i in range(len(player_positions)):
            reference_player_1 = self._create_reference_player(1, player_positions[i][1])
            reference_player_2 = self._create_reference_player(2, player_positions[i][2])
            reference_ball = self._create_reference_ball(1, ball_positions[i][1])
            reference_frame = ReferenceFrame(
                frame_number=i,
                near_player_id=near_player,
                far_player_id=far_player,
                player_1=reference_player_1,
                player_2=reference_player_2,
                ball=reference_ball
            )
            reference_frames.append(reference_frame)
        return reference_frames

    def _create_reference_player(self, player_id, bounding_box: Tuple[int, int, int, int]) -> ReferencePlayer:
        """
        Create a reference player from the bounding box.
        
        Args:
            bounding_box: Bounding box of the player
        """
        x1, y1, x2, y2 = bounding_box
        original_coordinate_bottom_center = ((x1 + x2) // 2, y2)
        reference_coordinate = self.get_reference_coordinate(original_coordinate_bottom_center)
        reference_player = ReferencePlayer(
            player_id=player_id,
            original_bounding_box=BoundingBox(x1, y1, x2, y2),
            reference_coordinate=reference_coordinate,
            action=PLAYER_RUN,
            net_clearance=0
        )
        return reference_player

    def _create_reference_ball(self, ball_id, bounding_box: Tuple[int, int, int, int]) -> ReferenceBall:
        """
        Create a reference ball from the bounding box.
        
        Args:
            bounding_box: Bounding box of the ball
        """
        x1, y1, x2, y2 = bounding_box
        original_coordinate_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        reference_ball = ReferenceBall(
            ball_id=ball_id,
            original_bounding_box=BoundingBox(x1, y1, x2, y2),
            reference_coordinate=self.get_reference_coordinate(original_coordinate_center),
            height_meters=0,
            horizontal_velocity=0,
            vertical_velocity=0,
            action=BALL_IN_FLIGHT
        )
        return reference_ball

    def _computer_far_hit_near_bounce(self, hits_and_bounces: List[int], hit_bounce_index: int, frame_count: int, fps: int) -> None:
        hitting_player = self.reference_frames[hits_and_bounces[hit_bounce_index]].get_far_player()
        hitting_player.action = PLAYER_HIT
        hitting_ball = self.reference_frames[hits_and_bounces[hit_bounce_index]].ball
        hitting_ball.action = BALL_FAR_HIT
        start_coordinate = (hitting_ball.reference_coordinate[0], int(hitting_player.reference_coordinate[1] + hitting_player.original_bounding_box.width_height_ratio * self.pixel_to_meter_ratio))
        bouncing_ball = self.reference_frames[hits_and_bounces[hit_bounce_index + 1]].ball
        bouncing_ball.action = BALL_NEAR_BOUNCE
        end_coordinate = bouncing_ball.reference_coordinate
        calculated_coordinates, calculated_horizontal_velocities = self._compute_horizontal_flight_path(start_coordinate, end_coordinate, frame_count, fps)
        self._replace_ball_coordinates_and_horozontal_velocity(hits_and_bounces[hit_bounce_index], calculated_coordinates, calculated_horizontal_velocities)

        hitting_player_box = self._move_player_box_down(hitting_player.original_bounding_box)
        hitting_ball_box = hitting_ball.original_bounding_box
        hitting_ball_player_ratio = (hitting_player_box.y2 - hitting_ball_box.y2) / (hitting_player_box.y2 - hitting_player_box.y1)
        ball_hit_height = round(self._get_player_height(hitting_player_box) * hitting_ball_player_ratio, 2)
        ball_heights, vertical_velocities = self._compute_vertical_flight_path_hit(frame_count, fps, ball_hit_height, 0.0)
        net_clearance = self._replace_ball_heights_and_vertical_velocities(hits_and_bounces[hit_bounce_index], ball_heights, vertical_velocities)
        hitting_player.net_clearance = net_clearance

    def _computer_near_bounce_near_hit(self, hits_and_bounces: List[int], hit_bounce_index: int, frame_count: int, fps: int) -> None:
        hitting_player = self.reference_frames[hits_and_bounces[hit_bounce_index + 1]].get_near_player()
        hitting_player.action = PLAYER_HIT
        bouncing_ball = self.reference_frames[hits_and_bounces[hit_bounce_index]].ball
        bouncing_ball.action = BALL_NEAR_BOUNCE
        start_coordinate = bouncing_ball.reference_coordinate
        hitting_ball = self.reference_frames[hits_and_bounces[hit_bounce_index + 1]].ball
        hitting_ball.action = BALL_NEAR_HIT
        end_coordinate = (hitting_ball.reference_coordinate[0], int(hitting_player.reference_coordinate[1] - 0.5 * self.pixel_to_meter_ratio))
        calculated_coordinates, calculated_horizontal_velocities = self._compute_horizontal_flight_path(start_coordinate, end_coordinate, frame_count, fps)
        self._replace_ball_coordinates_and_horozontal_velocity(hits_and_bounces[hit_bounce_index], calculated_coordinates, calculated_horizontal_velocities)

        hitting_player_box = self._move_player_box_up(hitting_player.original_bounding_box)
        hitting_ball_box = hitting_ball.original_bounding_box
        hitting_ball_player_ratio = (hitting_player_box.y2 - hitting_ball_box.y2) / (hitting_player_box.y2 - hitting_player_box.y1)
        ball_bounce_height = round(self._get_player_height(hitting_player_box) * hitting_ball_player_ratio, 2)
        ball_heights, vertical_velocities = self._compute_vertical_flight_path_bounce(frame_count, fps, 0, ball_bounce_height)
        net_clearance = self._replace_ball_heights_and_vertical_velocities(hits_and_bounces[hit_bounce_index], ball_heights, vertical_velocities)
        hitting_player.net_clearance = net_clearance

    def _computer_near_hit_far_bounce(self, hits_and_bounces: List[int], hit_bounce_index: int, frame_count: int, fps: int) -> None:
        hitting_player = self.reference_frames[hits_and_bounces[hit_bounce_index]].get_near_player()
        hitting_player.action = PLAYER_HIT
        hitting_ball = self.reference_frames[hits_and_bounces[hit_bounce_index]].ball
        hitting_ball.action = BALL_NEAR_HIT
        start_coordinate = (hitting_ball.reference_coordinate[0], int(hitting_player.reference_coordinate[1] - hitting_player.original_bounding_box.width_height_ratio * self.pixel_to_meter_ratio))
        bouncing_ball = self.reference_frames[hits_and_bounces[hit_bounce_index + 1]].ball
        bouncing_ball.action = BALL_FAR_BOUNCE
        end_coordinate = bouncing_ball.reference_coordinate
        calculated_coordinates, calculated_horizontal_velocities = self._compute_horizontal_flight_path(start_coordinate, end_coordinate, frame_count, fps)
        self._replace_ball_coordinates_and_horozontal_velocity(hits_and_bounces[hit_bounce_index], calculated_coordinates, calculated_horizontal_velocities)

        player_box = self._move_player_box_up(hitting_player.original_bounding_box)
        ball_box = hitting_ball.original_bounding_box
        ball_player_ratio = (player_box.y2 - ball_box.y2) / (player_box.y2 - player_box.y1)
        ball_hit_height = round(self._get_player_height(player_box) * ball_player_ratio, 2)
        ball_heights, vertical_velocities = self._compute_vertical_flight_path_hit(frame_count, fps, ball_hit_height, 0.0)
        net_clearance = self._replace_ball_heights_and_vertical_velocities(hits_and_bounces[hit_bounce_index], ball_heights, vertical_velocities)
        hitting_player.net_clearance = net_clearance

    def _computer_far_bounce_far_hit(self, hits_and_bounces: List[int], hit_bounce_index: int, frame_count: int, fps: int) -> None:
        hitting_player = self.reference_frames[hits_and_bounces[hit_bounce_index + 1]].get_far_player()
        hitting_player.action = PLAYER_HIT
        bouncing_ball = self.reference_frames[hits_and_bounces[hit_bounce_index]].ball
        bouncing_ball.action = BALL_FAR_BOUNCE
        start_coordinate = bouncing_ball.reference_coordinate
        hitting_ball = self.reference_frames[hits_and_bounces[hit_bounce_index + 1]].ball
        hitting_ball.action = BALL_FAR_HIT
        end_coordinate = (hitting_ball.reference_coordinate[0], int(hitting_player.reference_coordinate[1] + 0.5 * self.pixel_to_meter_ratio))
        calculated_coordinates, calculated_horizontal_velocities = self._compute_horizontal_flight_path(start_coordinate, end_coordinate, frame_count, fps)
        self._replace_ball_coordinates_and_horozontal_velocity(hits_and_bounces[hit_bounce_index], calculated_coordinates, calculated_horizontal_velocities)

        hitting_player_box = self._move_player_box_down(hitting_player.original_bounding_box)
        hitting_ball_box = hitting_ball.original_bounding_box
        hitting_ball_player_ratio = (hitting_player_box.y2 - hitting_ball_box.y2) / (hitting_player_box.y2 - hitting_player_box.y1)
        ball_bounce_height = round(self._get_player_height(hitting_player_box) * hitting_ball_player_ratio, 2)
        ball_heights, vertical_velocities = self._compute_vertical_flight_path_bounce(frame_count, fps, 0, ball_bounce_height)
        net_clearance = self._replace_ball_heights_and_vertical_velocities(hits_and_bounces[hit_bounce_index], ball_heights, vertical_velocities)
        hitting_player.net_clearance = net_clearance

    def _compute_horizontal_flight_path(self, 
                                       start_coordinate: Tuple[int, int], 
                                       end_coordinate: Tuple[int, int], 
                                       frame_count: int, 
                                       fps: float) -> Tuple[List[Tuple[int, int]], List[float]]:
        distance_meters = get_distance_between_points(start_coordinate, end_coordinate) / self.pixel_to_meter_ratio
        time_seconds = frame_count / fps
        initial_horizontal_velocity = round(get_initial_horizontal_velocity(distance_meters, time_seconds), 2)

        calculated_coordinates = [start_coordinate]
        calculated_horizontal_velocities = [initial_horizontal_velocity]

        # Calculate coordinates for intermediate frames
        for i in range(1, frame_count):
            time_at_frame = i / fps
            distance_at_frame = get_horizontal_distance_by_time(initial_horizontal_velocity, time_at_frame)
            ratio = distance_at_frame / distance_meters if distance_meters > 0 else 0
            x = start_coordinate[0] + ratio * (end_coordinate[0] - start_coordinate[0])
            y = start_coordinate[1] + ratio * (end_coordinate[1] - start_coordinate[1])
            calculated_coordinates.append((int(x), int(y)))
            calculated_horizontal_velocities.append(round(get_horizontal_velocity_by_time(initial_horizontal_velocity, time_at_frame), 2))

        return calculated_coordinates, calculated_horizontal_velocities

    def _replace_ball_coordinates_and_horozontal_velocity(self,
                                start_frame_number: int,
                                calculated_coordinates: List[Tuple[int, int]],
                                calculated_horizontal_velocities: List[float]) -> None:
        for j, coordinate in enumerate(calculated_coordinates):
            frame_number = start_frame_number + j
            self.reference_frames[frame_number].ball.reference_coordinate = coordinate
            self.reference_frames[frame_number].ball.horizontal_velocity = calculated_horizontal_velocities[j]

    def _move_player_box_down(self, player_box: BoundingBox) -> BoundingBox:
        reference_top_left = self.get_reference_coordinate((player_box.x1, player_box.y1))
        reference_bottom_right = self.get_reference_coordinate((player_box.x2, player_box.y2))
        reference_top_left = (reference_top_left[0], round(reference_top_left[1] + player_box.width_height_ratio * self.pixel_to_meter_ratio))
        reference_bottom_right = (reference_bottom_right[0], round(reference_bottom_right[1] + player_box.width_height_ratio * self.pixel_to_meter_ratio))
        x1, y1 = self.get_original_coordinate(reference_top_left)
        x2, y2 = self.get_original_coordinate(reference_bottom_right)
        return BoundingBox(x1, y1, x2, y2)

    def _move_player_box_up(self, player_box: BoundingBox) -> BoundingBox:
        reference_top_left = self.get_reference_coordinate((player_box.x1, player_box.y1))
        reference_bottom_right = self.get_reference_coordinate((player_box.x2, player_box.y2))
        reference_top_left = (reference_top_left[0], round(reference_top_left[1] - player_box.width_height_ratio * self.pixel_to_meter_ratio))
        reference_bottom_right = (reference_bottom_right[0], round(reference_bottom_right[1] - player_box.width_height_ratio * self.pixel_to_meter_ratio))
        x1, y1 = self.get_original_coordinate(reference_top_left)
        x2, y2 = self.get_original_coordinate(reference_bottom_right)
        return BoundingBox(x1, y1, x2, y2)

    def _get_player_height(self, player_box: BoundingBox) -> float:
        """
        Compute the height of the player based on the bounding box and 2 court base lines.
             /-----\
            /     / \
           /     /   \
          /-----/-----\
         /     /       \
        /-----/---------\
        Use the top and bottom base lines we can find out the middle line width in pixels where the player stands.
        Since we know the width of the doubles line in meters, we can calculate the player width in meters.
        Using the player width in meters and the player bounding box, we can calculate the player height in meters.
        """
        top_base_line_width = self.homographied_keypoints[1][0] - self.homographied_keypoints[0][0]
        bottom_base_line_width = self.homographied_keypoints[3][0] - self.homographied_keypoints[2][0]
        base_line_diff = bottom_base_line_width - top_base_line_width
        court_height = self.homographied_keypoints[2][1] - self.homographied_keypoints[0][1]
        player_to_top_base_line = player_box.y2 - self.homographied_keypoints[0][1]
        player_line_width = (player_to_top_base_line / court_height) * base_line_diff + top_base_line_width
        player_width_meters = (player_box.width / player_line_width) * DOUBLES_LINE_WIDTH
        player_height_meters = round((player_box.height / player_box.width) * player_width_meters, 2)
        print(f'Player height: {player_height_meters} m')
        return player_height_meters

    def _compute_vertical_flight_path_hit(self, frame_count: int, fps: float, initial_height: float, final_height: float) -> Tuple[List[float], List[float]]:
        """
        Compute the vertical flight path of the ball.
        
        Args:
            frame_number: Frame number of the hit
            fps: Frames per second of the video
            ball_hit_height: Height of the ball at the hit point
            
        Returns:
            Tuple of vertical distances and vertical velocities
        """
        time_seconds = frame_count / fps
        initial_vertical_velocity = round(get_initial_vertical_velocity_hit(initial_height, time_seconds), 2)
        vertical_distances, vertical_velocities = get_vertical_distances_and_velocities(
            initial_vertical_velocity, 
            initial_height, 
            time_seconds / frame_count)
        
        return vertical_distances, vertical_velocities

    def _compute_vertical_flight_path_bounce(self, frame_count: int, fps: float, initial_height: float, final_height: float) -> Tuple[List[float], List[float]]:
        """
        Compute the vertical flight path of the ball.
        
        Args:
            frame_number: Frame number of the hit
            fps: Frames per second of the video
            ball_hit_height: Height of the ball at the hit point
            
        Returns:
            Tuple of vertical distances and vertical velocities
        """
        time_seconds = frame_count / fps
        initial_vertical_velocity = round(get_initial_vertical_velocity_bounce(final_height, time_seconds), 2)
        vertical_distances, vertical_velocities = get_vertical_distances_and_velocities(
            initial_vertical_velocity, 
            initial_height, 
            time_seconds / frame_count)
        
        return vertical_distances, vertical_velocities

    def _replace_ball_heights_and_vertical_velocities(self,
                                start_frame_number: int,
                                ball_heights: List[float],
                                vertical_velocities: List[float]) -> float:
        balls: List[ReferenceBall] = []
        for j, height in enumerate(ball_heights):
            frame_number = start_frame_number + j
            ball = self.reference_frames[frame_number].ball
            balls.append(ball)
            ball.height_meters = height
            ball.vertical_velocity = vertical_velocities[j]
            # print(f'Frame number: {frame_number}, Ball height: {ball.height_meters}, Velocity: {ball.vertical_velocity}')

        # Calculate net clearance
        net_clearance = 0
        balls.sort(key=lambda ball: ball.reference_coordinate[1])
        for i in range(len(balls)):
            if balls[i].reference_coordinate[1] < self.canvas_depth / 2:
                continue
            if balls[i].reference_coordinate[1] == self.canvas_depth / 2:
                net_clearance = balls[i].height_meters
                break
            ball_1 = balls[i - 1]
            ball_2 = balls[i]
            y_ratio = (ball_2.reference_coordinate[1] - self.canvas_depth / 2) / (ball_2.reference_coordinate[1] - ball_1.reference_coordinate[1])
            ball_height_diff = ball_2.height_meters - ball_1.height_meters
            net_clearance = round(ball_2.height_meters - ball_height_diff * y_ratio, 2)
            break
        print(f'Ball is on the net: {net_clearance} m')
        return net_clearance

    def _get_net_clearance(self, calculated_coordinates: List[Tuple[int, int, float, float]]) -> None:
        """
        Calculate the net clearance for the ball trajectory.
        
        Args:
            calculated_coordinates: List of calculated ball coordinates
        """
        if calculated_coordinates is None:
            return
        if len(calculated_coordinates[0]) == 2:
            print('No net clearance, its a bounce and hit')
            return
        coordinates = copy.deepcopy(calculated_coordinates)
        coordinates.sort(key=lambda x: x[1])
        for i in range(len(coordinates)):
            if coordinates[i][1] < self.canvas_depth / 2:
                continue
            if coordinates[i][1] == self.canvas_depth / 2:
                print(f'Ball is on the net: {coordinates[i][2]} m')
                return
            coordinate1 = coordinates[i - 1]
            coordinate2 = coordinates[i]
            y_ratio = (coordinate2[1] - self.canvas_depth / 2) / (coordinate2[1] - coordinate1[1])
            ball_height_diff = coordinate2[2] - coordinate1[2]
            ball_height_at_net = coordinate2[2] - ball_height_diff * y_ratio
            print(f'Net clearance: {round(ball_height_at_net, 2)} m')
            break

    def draw(self, input_frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Draw the reference court, players, and ball on the input frames.
        
        Args:
            input_frames: List of input frames
            
        Returns:
            List of output frames with visualization
        """
        output_frames = []
        
        for frame_number, frame in enumerate(input_frames):
            # Draw reference court canvas and court lines
            frame = self._draw_canvas(frame)
            frame = self._draw_court(frame)
            
            # Draw players and ball
            self._draw_player_coordinates(frame, 
                                          self.reference_frames[frame_number].far_player_id, 
                                          self.reference_frames[frame_number].player_1, 
                                          10, 
                                          color=(0, 0, 255))  # Near player
            self._draw_player_coordinates(frame, 
                                          self.reference_frames[frame_number].far_player_id, 
                                          self.reference_frames[frame_number].player_2, 
                                          10, 
                                          color=(255, 0, 0))  # Far player
            self._draw_ball_coordinates(frame, 
                                        self.reference_frames[frame_number].ball, 
                                        5, 
                                        color=(0, 255, 0))    # Ball
            
            output_frames.append(frame)
            
        return output_frames

    def _draw_player_coordinates(self,
                                frame: np.ndarray,
                                far_player_id: int,
                                player: ReferencePlayer,
                                size: int,
                                color: Tuple[int, int, int]) -> None:
        x = int(player.reference_coordinate[0] + self.canvas_x1)
        y = int(player.reference_coordinate[1] + self.canvas_y1)
        cv2.circle(frame, (x, y), size, color, -1)
        if player.player_id == far_player_id:
            cv2.putText(frame, f"{player.player_id}", (self.canvas_x1 + 10, self.canvas_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"{player.player_id}", (self.canvas_x1 + 10, self.canvas_y2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    def _draw_ball_coordinates(self, 
                        frame: np.ndarray, 
                        ball: ReferenceBall,
                        size: int, 
                        color: Tuple[int, int, int] = (0, 255, 0)) -> None:
        """
        Draw coordinates on the frame.
        
        Args:
            frame: Frame to draw on
            coordinate: x-y values
            size: Size of the circle
            color: Color of the circle
        """
        x = int(ball.reference_coordinate[0] + self.canvas_x1)
        y = int(ball.reference_coordinate[1] + self.canvas_y1)
        cv2.circle(frame, (x, y), size, color, -1)

        cv2.putText(
            frame, f"h: {ball.height_meters}", (x + 10, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"v: {get_hypotenuse(ball.horizontal_velocity, ball.vertical_velocity)}", (x + 10, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    def _draw_court(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw court lines and key points on the frame.
        
        Args:
            frame: Frame to draw on
            
        Returns:
            Frame with court lines and key points
        """
        # Draw lines
        for line in self.court_lines:
            x1 = int(self.keypoints[line[0]][0] + self.canvas_x1)
            y1 = int(self.keypoints[line[0]][1] + self.canvas_y1)
            x2 = int(self.keypoints[line[1]][0] + self.canvas_x1)
            y2 = int(self.keypoints[line[1]][1] + self.canvas_y1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

        # Draw center line
        center_x = int((self.keypoints[0][0] + self.keypoints[1][0]) / 2 + self.canvas_x1)
        
        # Top center line
        top_y1 = int(self.keypoints[0][1] + self.canvas_y1)
        top_y2 = int(top_y1 + CENTER_LINE_DEPTH * self.pixel_to_meter_ratio)
        cv2.line(frame, (center_x, top_y1), (center_x, top_y2), (0, 0, 0), 2)
        
        # Bottom center line
        bottom_y1 = int(self.keypoints[2][1] + self.canvas_y1)
        bottom_y2 = int(bottom_y1 - CENTER_LINE_DEPTH * self.pixel_to_meter_ratio)
        cv2.line(frame, (center_x, bottom_y1), (center_x, bottom_y2), (0, 0, 0), 2)

        # Draw net
        net_x1 = int(self.court_x1 + self.canvas_x1)
        net_x2 = int(self.court_x2 + self.canvas_x1)
        net_y = int((self.canvas_y1 + self.canvas_y2) / 2)
        cv2.line(frame, (net_x1, net_y), (net_x2, net_y), (255, 0, 0), 2)

        # Draw key points
        for keypoint in self.keypoints:
            x = int(keypoint[0] + self.canvas_x1)
            y = int(keypoint[1] + self.canvas_y1)
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

        return frame

    def _draw_canvas(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw a semi-transparent canvas on the frame.
        
        Args:
            frame: Frame to draw on
            
        Returns:
            Frame with canvas
        """
        # Create a blank image with the same size as the input frame
        shapes = np.zeros_like(frame, np.uint8)
        
        # Draw a filled white rectangle for the canvas
        cv2.rectangle(
            shapes, 
            (int(self.canvas_x1), int(self.canvas_y1)), 
            (int(self.canvas_x2), int(self.canvas_y2)), 
            (255, 255, 255), 
            cv2.FILLED
        )
        
        output_frame = frame.copy()
        
        # Apply transparent overlay
        alpha = 0.5
        mask = shapes.astype(bool)
        output_frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return output_frame
