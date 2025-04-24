import cv2
import numpy as np
import math
import copy
import pandas as pd
from scipy.spatial import distance
from typing import List, Dict, Tuple, Optional, Any
from .utils import get_bottom_line_center_point, \
    get_bounding_box_center_point, \
    get_distance_between_points, \
    get_initial_horizontal_velocity, \
    get_horizontal_distance_by_time, \
    get_horizontal_velocity_by_time, \
    get_initial_vertical_velocity, \
    get_vertical_distances_and_velocities, \
    get_hypotenuse, \
    get_player_height
from .constants import DOUBLES_LINE_WIDTH, DOUBLES_ALLEY_WIDTH, RUN_BACK_DEPTH, \
    SIDE_RUN_WIDTH, HALF_COURT_DEPTH, NO_MANS_LAND_DEPTH, REFERENCE_COURT_MARGIN_X, \
    REFERENCE_COURT_MARGIN_Y, CENTER_LINE_DEPTH


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
        self.canvas_depth = ((HALF_COURT_DEPTH + RUN_BACK_DEPTH) * 2) * self.pixel_to_meter_ratio

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
        self.player_coordinates = None
        self.ball_coordinates = None
        
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

    def homograph_keypoints(self, given_keypoints: List[Optional[Tuple[int, int]]]) -> List[Tuple[int, int]]:
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
            matrix, _ = cv2.findHomography(
                np.float32(reference_configuration), 
                np.float32(given_configuration),
                # method=cv2.LMEDS
                method=cv2.RANSAC
            )
            
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
        homographied_keypoints = [tuple(map(int, point)) for point in best_keypoints.reshape(-1, 2)]
        
        # Preserve detected keypoints where available
        for i in range(len(homographied_keypoints)):
            if given_keypoints[i] is not None:
                homographied_keypoints[i] = given_keypoints[i]
                
        return homographied_keypoints

    def convert_player_coordinates(self, player_positions: List[Dict[int, List]]) -> None:
        """
        Convert player bounding box coordinates to reference coordinates.
        
        Args:
            player_positions: List of dictionaries containing player bounding boxes
        """
        self.player_coordinates = self.convert_to_reference_coordinates(player_positions)

    def convert_to_reference_coordinates(self, positions_all_frames: List[Dict[int, Any]]) -> List[Dict[int, Tuple[int, int]]]:
        """
        Convert original coordinates to reference coordinates for all frames.
        
        Args:
            positions_all_frames: List of dictionaries containing positions per frame
            
        Returns:
            List of dictionaries containing reference coordinates per frame
        """
        reference_coordinates = []
        
        for positions_per_frame in positions_all_frames:
            reference_coordinates_per_frame = {}
            
            for track_id, bounding_box in positions_per_frame.items():
                original_coordinate = get_bottom_line_center_point(bounding_box)
                reference_coordinate = self.get_reference_coordinate(original_coordinate)
                reference_coordinates_per_frame[track_id] = reference_coordinate
                
            reference_coordinates.append(reference_coordinates_per_frame)
            
        return reference_coordinates

    def get_reference_coordinate(self, original_coordinate: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert original coordinate to reference coordinate.
        
        Args:
            original_coordinate: Original coordinate in the image
            
        Returns:
            Reference coordinate
        """
        original_coordinate = np.array(original_coordinate, dtype=np.float32).reshape((-1, 1, 2))
        reference_coordinate = cv2.perspectiveTransform(original_coordinate, self.inverse_homography_matrix)
        reference_coordinate = reference_coordinate[0][0]
        return int(reference_coordinate[0]), int(reference_coordinate[1])

    def get_original_coordinate(self, reference_coordinate: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert reference coordinate to original coordinate.
        
        Args:
            reference_coordinate: Reference coordinate
            
        Returns:
            Original coordinate in the image
        """
        reference_coordinate = np.array(reference_coordinate, dtype=np.float32).reshape((-1, 1, 2))
        original_coordinate = cv2.perspectiveTransform(reference_coordinate, self.homography_matrix)
        original_coordinate = original_coordinate[0][0]
        return int(original_coordinate[0]), int(original_coordinate[1])

    def convert_ball_coordinates(self, 
                                player_positions: List[Dict[int, List]], 
                                near_player: int, 
                                far_player: int, 
                                ball_positions: List[Dict[int, List]], 
                                hits_and_bounces: List[Optional[int]], 
                                fps: float) -> None:
        """
        Convert detected ball coordinates to reference coordinates with calculated flight path.
        
        Args:
            player_positions: List of dictionaries containing player bounding boxes
            near_player: Track ID of near player
            far_player: Track ID of far player
            ball_positions: List of dictionaries containing ball bounding boxes
            hits_and_bounces: List of frame numbers of hits and bounces
            fps: Frames per second of the video
        """
        self.ball_coordinates = self.convert_to_reference_coordinates(ball_positions)
        
        i = 0
        # hits_and_bounces in the order of far_hit, near_bounce, near_hit, far_bounce
        while i < len(hits_and_bounces) - 1:
            # Skip if hit or bounce is missing
            if hits_and_bounces[i] is None or hits_and_bounces[i + 1] is None:
                i += 1
                continue
                
            print(f'Processing hit and bounce: {hits_and_bounces[i]} - {hits_and_bounces[i + 1]}')
            frame_count = hits_and_bounces[i + 1] - hits_and_bounces[i]
            
            ball_point_1 = get_bounding_box_center_point(ball_positions[hits_and_bounces[i]][1])
            ball_point_2 = get_bounding_box_center_point(ball_positions[hits_and_bounces[i + 1]][1])
            
            player_box = self._get_player_box(player_positions, hits_and_bounces, i, far_player, near_player)
            start_point, end_point = self._get_start_and_end_points(player_box, i, ball_point_1, ball_point_2)
            
            ball_hit_height = self._get_ball_hit_height(player_box, i, ball_point_1)
            calculated_coordinates = self._calculate_flight_path(start_point, end_point, frame_count, fps, ball_hit_height)
            self._replace_ball_coordinates(hits_and_bounces[i], calculated_coordinates)
            
            i += 1
        
        df = pd.DataFrame({
            'ball_position': self.ball_coordinates,
            'frame': range(len(self.ball_coordinates)),
        })
        df.to_csv('reference_data_frame.csv', index=False)

    def _get_ball_hit_height(self, 
                             player_box: List[int], 
                             hits_and_bounces_index: int, 
                             ball_point: Tuple[int, int]) -> float:
        hit_bounce_pattern = hits_and_bounces_index % 4
        if hit_bounce_pattern == 1 or hit_bounce_pattern == 3:
            return 0

        # Calculate the height of the ball at the moment of hit or bounce       
        player_ratio = (player_box[2] - player_box[0]) / (player_box[3] - player_box[1])
        player_height = get_player_height(player_ratio)
        if hit_bounce_pattern == 0:  # Far hit and bounce
            # Move player feet 1 meter down
            player_feet_point = (player_box[2], player_box[3])
            player_feet_reference_point = self.get_reference_coordinate(player_feet_point)
            player_hitting_reference_point = (player_feet_reference_point[0], player_feet_reference_point[1] + 1 * self.pixel_to_meter_ratio)
            player_hitting_point = self.get_original_coordinate(player_hitting_reference_point)
            ball_player_ratio = (player_box[3] - ball_point[1]) / (player_box[3] - player_box[1])
            if ball_player_ratio < 0:
                ball_player_ratio = 0.01
            ball_hit_height = round(player_height * ball_player_ratio, 2)
            print(f'Far hit and bounce, {ball_hit_height} m, player_height: {player_height} m')
            return ball_hit_height
        else:  # Near hit and bounce
            # Move player feet 1 meter up
            player_feet_point = (player_box[2], player_box[3])
            player_feet_reference_point = self.get_reference_coordinate(player_feet_point)
            player_hitting_reference_point = (player_feet_reference_point[0], player_feet_reference_point[1] - 1 * self.pixel_to_meter_ratio)
            player_hitting_point = self.get_original_coordinate(player_hitting_reference_point)
            ball_player_ratio = (player_hitting_point[1] - ball_point[1]) / (player_box[3] - player_box[1])
            if ball_player_ratio < 0:
                ball_player_ratio = 0.01
            ball_hit_height = round(player_height * ball_player_ratio, 2)
            print(f'Near hit and bounce, {ball_hit_height} m, player_height: {player_height} m')
            return ball_hit_height

    def _get_player_box(self, 
                      player_positions: List[Dict[int, List]], 
                      hits_and_bounces: List[Optional[int]], 
                      index: int, 
                      far_player: int, 
                      near_player: int) -> List[int]:
        """
        Get the player bounding box for the hitting player.
        
        Args:
            player_positions: List of dictionaries containing player bounding boxes
            hits_and_bounces: List of frame numbers of hits and bounces
            index: Index in hits_and_bounces list
            far_player: Track ID of far player
            near_player: Track ID of near player
            
        Returns:
            Bounding box of the player
        """
        hit_bounce_pattern = index % 4
        
        if hit_bounce_pattern == 0:  # Far hit and bounce
            frame = hits_and_bounces[index]
            player_id = far_player
        elif hit_bounce_pattern == 1:  # Near bounce and hit
            frame = hits_and_bounces[index + 1]
            player_id = near_player
        elif hit_bounce_pattern == 2:  # Near hit and bounce
            frame = hits_and_bounces[index]
            player_id = near_player
        else:  # Far bounce and hit
            frame = hits_and_bounces[index + 1]
            player_id = far_player
            
        return [int(x) for x in player_positions[frame][player_id]]

    def _get_start_and_end_points(self, 
                                player_box: List[int], 
                                hits_and_bounces_index: int, 
                                ball_point_1: Tuple[int, int], 
                                ball_point_2: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Get the start and end points of a bounce-hit or hit-bounce.
        
        Args:
            player_box: Bounding box of the player
            hits_and_bounces_index: Index in hits_and_bounces list
            ball_point_1: Coordinate of the first ball point
            ball_point_2: Coordinate of the second ball point
            
        Returns:
            Tuple of start and end points in reference coordinates
        """
        if hits_and_bounces_index % 2 == 0:
            # Hit and bounce: Adjust hit point to player's feet
            hit_point = (ball_point_1[0], player_box[3])
            start_point = self.get_reference_coordinate(hit_point)
            end_point = self.get_reference_coordinate(ball_point_2)
        else:
            # Bounce and hit: Adjust hit point to player's feet
            hit_point = (ball_point_2[0], player_box[3])
            start_point = self.get_reference_coordinate(ball_point_1)
            end_point = self.get_reference_coordinate(hit_point)
            
        return start_point, end_point

    def _calculate_flight_path(self, 
                             start_point: Tuple[int, int], 
                             end_point: Tuple[int, int], 
                             frame_count: int, 
                             fps: float,
                             ball_hit_height: float = 0) -> List[Tuple[int, int]]:
        """
        Calculate the ball's flight path using trajectory equation with air drag.
        
        Args:
            start_point: Start point of the flight path
            end_point: End point of the flight path
            frame_count: Number of frames between start and end points
            fps: Frames per second of the video
            
        Returns:
            List of calculated ball coordinates
        """
        distance_meters = get_distance_between_points(start_point, end_point) / self.pixel_to_meter_ratio
        time_seconds = frame_count / fps
        initial_horizontal_velocity = get_initial_horizontal_velocity(distance_meters, time_seconds)
        print(f'Distance: {distance_meters}, Time: {time_seconds}, Initial horizontal velocity: {initial_horizontal_velocity} m/s')

        calculated_coordinates = [start_point]
        vertical_distances = None
        vertical_velocities = None

        if ball_hit_height > 0:
            initial_vertical_velocity = get_initial_vertical_velocity(ball_hit_height, time_seconds)
            vertical_distances, vertical_velocities = get_vertical_distances_and_velocities(
                initial_vertical_velocity, 
                ball_hit_height, 
                time_seconds / frame_count)

            initial_velocity = get_hypotenuse(initial_horizontal_velocity, initial_vertical_velocity)

            calculated_coordinates = [(start_point[0], start_point[1], ball_hit_height, round(initial_velocity, 2))]
        
        # Calculate coordinates for intermediate frames
        for i in range(1, frame_count):
            time_at_frame = i / fps
            distance_at_time = get_horizontal_distance_by_time(initial_horizontal_velocity, time_at_frame)
            ratio = distance_at_time / distance_meters if distance_meters > 0 else 0
            
            x = start_point[0] + ratio * (end_point[0] - start_point[0])
            y = start_point[1] + ratio * (end_point[1] - start_point[1])
            if ball_hit_height > 0:
                h = vertical_distances[i]
                horizontal_velocity = get_horizontal_velocity_by_time(initial_velocity, time_at_frame)
                v = get_hypotenuse(horizontal_velocity, vertical_velocities[i])
                calculated_coordinates.append((int(x), int(y), round(h, 2), round(v, 2)))
            else:
                calculated_coordinates.append((int(x), int(y)))
        
        self.get_net_clearance(calculated_coordinates)

        return calculated_coordinates

    def get_net_clearance(self, calculated_coordinates: List[Tuple[int, int]]) -> None:
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

    def _replace_ball_coordinates(self, 
                                start_frame_number: int, 
                                calculated_coordinates: List[Tuple[int, int]]) -> None:
        """
        Replace detected ball coordinates with calculated coordinates.
        
        Args:
            start_frame_number: the frame number to start replacing
            calculated_coordinates: List of calculated ball coordinates
        """
        for j, coordinate in enumerate(calculated_coordinates):
            frame_number = start_frame_number + j
            self.ball_coordinates[frame_number] = {1: coordinate}

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
            self._draw_coordinates(frame, self.player_coordinates[frame_number][1], 10, color=(0, 0, 255))  # Near player
            self._draw_coordinates(frame, self.player_coordinates[frame_number][2], 10, color=(255, 0, 0))  # Far player
            self._draw_coordinates(frame, self.ball_coordinates[frame_number][1], 5, color=(0, 255, 0))     # Ball
            
            output_frames.append(frame)
            
        return output_frames

    def _draw_coordinates(self, 
                        frame: np.ndarray, 
                        coordinate: Tuple[int, int], 
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
        x = int(coordinate[0] + self.canvas_x1)
        y = int(coordinate[1] + self.canvas_y1)
        cv2.circle(frame, (x, y), size, color, -1)

        if len(coordinate) == 4:
            cv2.putText(
                frame, f"h: {coordinate[2]}", (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
            )
            cv2.putText(
                frame, f"v: {coordinate[3]}", (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
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
