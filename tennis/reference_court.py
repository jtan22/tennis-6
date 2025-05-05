import cv2
import numpy as np
import copy
import pandas as pd
from typing import List, Dict, Tuple

from tennis.bounding_box import BoundingBox
from tennis.reference_frame import ReferenceFrame
from tennis.reference_player import ReferencePlayer
from tennis.reference_ball import ReferenceBall

from .utils import (
    get_distance_between_points,
    get_initial_horizontal_velocity,
    get_horizontal_distance_by_time,
    get_horizontal_velocity_by_time,
    get_initial_vertical_velocity_hit,
    get_initial_vertical_velocity_bounce,
    get_vertical_distances_and_velocities,
    get_hypotenuse,
    get_homography_matrix,
    transform_coordinates,
)
from .constants import (
    BALL_FAR_BOUNCE, 
    DOUBLES_LINE_WIDTH, 
    REFERENCE_COURT_PIXEL_TO_METER_RATIO,
    CENTER_LINE_DEPTH, 
    BALL_FAR_HIT, 
    BALL_NEAR_BOUNCE,
    BALL_NEAR_HIT, 
    BALL_IN_FLIGHT, 
    PLAYER_RUNNING, 
    PLAYER_HITTING,
    REFERENCE_COURT_X1,
    REFERENCE_COURT_X2,
    REFERENCE_KEYPOINTS,
    REFERENCE_COURT_CANVAS_DEPTH,
    REFERENCE_COURT_CANVAS_WIDTH,
    REFERENCE_COURT_MARGIN_X,
    REFERENCE_COURT_MARGIN_Y,
    REFERENCE_COURT_LINES,
    NEAR_PLAYER_NAME,
    FAR_PLAYER_NAME,
)

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
    
    def __init__(self):
        self.reference_frames_path = 'analysis/reference_frames.csv'

        # Initialize member variables that will be set later
        self.homography_matrix: np.ndarray = np.empty((3, 3), dtype=np.float32)
        self.inverse_homography_matrix = np.empty((3, 3), dtype=np.float32)
        self.court_keypoints = []
        self.reference_frames = []
        
    def compute_reference_coordinates(self, 
                                court_keypoints: List[Tuple[int, int]],
                                near_player_positions: List[Tuple[int, int, int, int]], 
                                far_player_positions: List[Tuple[int, int, int, int]], 
                                ball_positions: List[Tuple[int, int, int, int]], 
                                hits_and_bounces: List[int | None], 
                                fps: int) -> None:
        self.court_keypoints = court_keypoints
        self.homography_matrix = get_homography_matrix(REFERENCE_KEYPOINTS, court_keypoints)
        self.inverse_homography_matrix = np.linalg.inv(self.homography_matrix)
        self.reference_frames = self._create_reference_frames(near_player_positions, far_player_positions, ball_positions)
        self._compute_ball_coordinates(hits_and_bounces, fps)
        df = pd.DataFrame({
            'frame_number': [frame.frame_number for frame in self.reference_frames],
            'near_player_original_box': [frame.near_player.original_bounding_box for frame in self.reference_frames],
            'near_player_reference_point': [frame.near_player.reference_coordinate for frame in self.reference_frames],
            'near_player_action': [frame.near_player.action for frame in self.reference_frames],
            'far_player_original_box': [frame.far_player.original_bounding_box for frame in self.reference_frames],
            'far_player_reference_point': [frame.far_player.reference_coordinate for frame in self.reference_frames],
            'far_player_action': [frame.far_player.action for frame in self.reference_frames],
            'ball_original_box': [frame.ball.original_bounding_box for frame in self.reference_frames],
            'ball_reference_point': [frame.ball.reference_coordinate for frame in self.reference_frames],
            'ball_height': [frame.ball.height_meters for frame in self.reference_frames],
            'ball_horizontal_velocity': [frame.ball.horizontal_velocity for frame in self.reference_frames],
            'ball_vertical_velocity': [frame.ball.vertical_velocity for frame in self.reference_frames],
            'ball_action': [frame.ball.action for frame in self.reference_frames],
        })
        df.to_csv(self.reference_frames_path, index=False)

    def _compute_ball_coordinates(self, hits_and_bounces: List[int | None], fps: int) -> None:
        i = 0
        # hits_and_bounces in the order of far_hit, near_bounce, near_hit, far_bounce
        while i < len(hits_and_bounces) - 1:
            # Skip if hit or bounce is missing
            hits_and_bounces_1: int = hits_and_bounces[i] if hits_and_bounces[i] is not None else -1 # type: ignore
            hits_and_bounces_2: int = hits_and_bounces[i + 1] if hits_and_bounces[i + 1] is not None else -1 # type: ignore
            if hits_and_bounces_1 < 0 or hits_and_bounces_2 < 0:
                i += 1
                continue
                
            print(f'Processing hit and bounce: {hits_and_bounces[i]} - {hits_and_bounces[i + 1]}')
            frame_count = hits_and_bounces_2 - hits_and_bounces_1

            hit_bounce_pattern = i % 4
            if hit_bounce_pattern == BALL_FAR_HIT:  # Far hit and bounce
                self._computer_far_hit_near_bounce(hits_and_bounces_1, hits_and_bounces_2, frame_count, fps)
            elif hit_bounce_pattern == BALL_NEAR_BOUNCE:  # Near bounce and hit
                self._computer_near_bounce_near_hit(hits_and_bounces_1, hits_and_bounces_2, frame_count, fps)
            elif hit_bounce_pattern == BALL_NEAR_HIT:  # Near hit and bounce
                self._computer_near_hit_far_bounce(hits_and_bounces_1, hits_and_bounces_2, frame_count, fps)
            else:  # Far bounce and hit
                self._computer_far_bounce_far_hit(hits_and_bounces_1, hits_and_bounces_2, frame_count, fps)

            i += 1

    def _create_reference_frames(self, 
                                near_player_positions: List[Tuple[int, int, int, int]], 
                                far_player_positions: List[Tuple[int, int, int, int]],
                                ball_positions: List[Tuple[int, int, int, int]]) -> List[ReferenceFrame]:
        reference_frames = []
        for i in range(len(ball_positions)):
            near_reference_player = self._create_reference_player(near_player_positions[i])
            far_reference_player = self._create_reference_player(far_player_positions[i])
            reference_ball = self._create_reference_ball(ball_positions[i])
            reference_frame = ReferenceFrame(
                frame_number=i,
                near_player=near_reference_player,
                far_player=far_reference_player,
                ball=reference_ball
            )
            reference_frames.append(reference_frame)
        return reference_frames

    def _create_reference_player(self, player_positions: Tuple[int, int, int, int]) -> ReferencePlayer:
        x1, y1, x2, y2 = player_positions
        original_coordinate_bottom_center = ((x1 + x2) // 2, y2)
        reference_coordinate = transform_coordinates([original_coordinate_bottom_center], self.inverse_homography_matrix)[0]
        reference_player = ReferencePlayer(
            original_bounding_box=BoundingBox(x1, y1, x2, y2),
            reference_coordinate=reference_coordinate,
            action=PLAYER_RUNNING,
        )
        return reference_player

    def _create_reference_ball(self, ball_positions: Tuple[int, int, int, int]) -> ReferenceBall:
        x1, y1, x2, y2 = ball_positions
        original_coordinate_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        reference_ball = ReferenceBall(
            original_bounding_box=BoundingBox(x1, y1, x2, y2),
            reference_coordinate=transform_coordinates([original_coordinate_center], self.inverse_homography_matrix)[0],
            height_meters=0,
            horizontal_velocity=0,
            vertical_velocity=0,
            action=BALL_IN_FLIGHT
        )
        return reference_ball

    def _computer_far_hit_near_bounce(self, hits_and_bounces_1: int, hits_and_bounces_2: int, frame_count: int, fps: int) -> None:
        hitting_player = self.reference_frames[hits_and_bounces_1].far_player
        hitting_player.action = PLAYER_HITTING
        hitting_ball = self.reference_frames[hits_and_bounces_1].ball
        hitting_ball.action = BALL_FAR_HIT
        start_coordinate = (
            hitting_ball.reference_coordinate[0], 
            int(hitting_player.reference_coordinate[1] + hitting_player.original_bounding_box.width_height_ratio * REFERENCE_COURT_PIXEL_TO_METER_RATIO))
        bouncing_ball = self.reference_frames[hits_and_bounces_2].ball
        bouncing_ball.action = BALL_NEAR_BOUNCE
        end_coordinate = bouncing_ball.reference_coordinate
        calculated_coordinates, calculated_horizontal_velocities = self._compute_horizontal_flight_path(
            start_coordinate, 
            end_coordinate, 
            frame_count, 
            fps)
        self._replace_ball_coordinates_and_horozontal_velocity(
            hits_and_bounces_1, 
            calculated_coordinates, 
            calculated_horizontal_velocities)

        hitting_player_box = self._move_player_box_down(hitting_player.original_bounding_box)
        hitting_ball_box = hitting_ball.original_bounding_box
        hitting_ball_player_ratio = (hitting_player_box.y2 - hitting_ball_box.y2) / (hitting_player_box.y2 - hitting_player_box.y1)
        ball_hit_height = round(self._get_player_height(hitting_player_box) * hitting_ball_player_ratio, 2)
        ball_heights, vertical_velocities = self._compute_vertical_flight_path_hit(frame_count, fps, ball_hit_height)
        self._replace_ball_heights_and_vertical_velocities(hits_and_bounces_1, ball_heights, vertical_velocities)

    def _computer_near_bounce_near_hit(self, hits_and_bounces_1: int, hits_and_bounces_2: int, frame_count: int, fps: int) -> None:
        hitting_player = self.reference_frames[hits_and_bounces_2].near_player
        hitting_player.action = PLAYER_HITTING
        bouncing_ball = self.reference_frames[hits_and_bounces_1].ball
        bouncing_ball.action = BALL_NEAR_BOUNCE
        start_coordinate = bouncing_ball.reference_coordinate
        hitting_ball = self.reference_frames[hits_and_bounces_2].ball
        hitting_ball.action = BALL_NEAR_HIT
        end_coordinate = (
            hitting_ball.reference_coordinate[0], 
            int(hitting_player.reference_coordinate[1] - 0.5 * REFERENCE_COURT_PIXEL_TO_METER_RATIO))
        calculated_coordinates, calculated_horizontal_velocities = self._compute_horizontal_flight_path(
            start_coordinate, 
            end_coordinate, 
            frame_count, 
            fps)
        self._replace_ball_coordinates_and_horozontal_velocity(
            hits_and_bounces_1, 
            calculated_coordinates, 
            calculated_horizontal_velocities)

        hitting_player_box = self._move_player_box_up(hitting_player.original_bounding_box)
        hitting_ball_box = hitting_ball.original_bounding_box
        hitting_ball_player_ratio = (hitting_player_box.y2 - hitting_ball_box.y2) / (hitting_player_box.y2 - hitting_player_box.y1)
        ball_bounce_height = round(self._get_player_height(hitting_player_box) * hitting_ball_player_ratio, 2)
        ball_heights, vertical_velocities = self._compute_vertical_flight_path_bounce(frame_count, fps, 0, ball_bounce_height)
        self._replace_ball_heights_and_vertical_velocities(hits_and_bounces_1, ball_heights, vertical_velocities)

    def _computer_near_hit_far_bounce(self, hits_and_bounces_1: int, hits_and_bounces_2: int, frame_count: int, fps: int) -> None:
        hitting_player = self.reference_frames[hits_and_bounces_1].near_player
        hitting_player.action = PLAYER_HITTING
        hitting_ball = self.reference_frames[hits_and_bounces_1].ball
        hitting_ball.action = BALL_NEAR_HIT
        start_coordinate = (
            hitting_ball.reference_coordinate[0], 
            int(hitting_player.reference_coordinate[1] - hitting_player.original_bounding_box.width_height_ratio * REFERENCE_COURT_PIXEL_TO_METER_RATIO))
        bouncing_ball = self.reference_frames[hits_and_bounces_2].ball
        bouncing_ball.action = BALL_FAR_BOUNCE
        end_coordinate = bouncing_ball.reference_coordinate
        calculated_coordinates, calculated_horizontal_velocities = self._compute_horizontal_flight_path(
            start_coordinate, 
            end_coordinate, 
            frame_count, 
            fps)
        self._replace_ball_coordinates_and_horozontal_velocity(
            hits_and_bounces_1, 
            calculated_coordinates, 
            calculated_horizontal_velocities)

        player_box = self._move_player_box_up(hitting_player.original_bounding_box)
        ball_box = hitting_ball.original_bounding_box
        ball_player_ratio = (player_box.y2 - ball_box.y2) / (player_box.y2 - player_box.y1)
        ball_hit_height = round(self._get_player_height(player_box) * ball_player_ratio, 2)
        ball_heights, vertical_velocities = self._compute_vertical_flight_path_hit(frame_count, fps, ball_hit_height)
        self._replace_ball_heights_and_vertical_velocities(hits_and_bounces_1, ball_heights, vertical_velocities)

    def _computer_far_bounce_far_hit(self, hits_and_bounces_1: int, hits_and_bounces_2: int, frame_count: int, fps: int) -> None:
        hitting_player = self.reference_frames[hits_and_bounces_2].far_player
        hitting_player.action = PLAYER_HITTING
        bouncing_ball = self.reference_frames[hits_and_bounces_1].ball
        bouncing_ball.action = BALL_FAR_BOUNCE
        start_coordinate = bouncing_ball.reference_coordinate
        hitting_ball = self.reference_frames[hits_and_bounces_2].ball
        hitting_ball.action = BALL_FAR_HIT
        end_coordinate = (
            hitting_ball.reference_coordinate[0], 
            int(hitting_player.reference_coordinate[1] + 0.5 * REFERENCE_COURT_PIXEL_TO_METER_RATIO))
        calculated_coordinates, calculated_horizontal_velocities = self._compute_horizontal_flight_path(
            start_coordinate, 
            end_coordinate, 
            frame_count, 
            fps)
        self._replace_ball_coordinates_and_horozontal_velocity(
            hits_and_bounces_1, 
            calculated_coordinates, 
            calculated_horizontal_velocities)

        hitting_player_box = self._move_player_box_down(hitting_player.original_bounding_box)
        hitting_ball_box = hitting_ball.original_bounding_box
        hitting_ball_player_ratio = (hitting_player_box.y2 - hitting_ball_box.y2) / (hitting_player_box.y2 - hitting_player_box.y1)
        ball_bounce_height = round(self._get_player_height(hitting_player_box) * hitting_ball_player_ratio, 2)
        ball_heights, vertical_velocities = self._compute_vertical_flight_path_bounce(frame_count, fps, 0, ball_bounce_height)
        self._replace_ball_heights_and_vertical_velocities(hits_and_bounces_1, ball_heights, vertical_velocities)

    def _compute_horizontal_flight_path(self, 
                                       start_coordinate: Tuple[int, int], 
                                       end_coordinate: Tuple[int, int], 
                                       frame_count: int, 
                                       fps: float) -> Tuple[List[Tuple[int, int]], List[float]]:
        distance_meters = get_distance_between_points(start_coordinate, end_coordinate) / REFERENCE_COURT_PIXEL_TO_METER_RATIO
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
        return self._move_player_box_vertically(player_box, 1)

    def _move_player_box_up(self, player_box: BoundingBox) -> BoundingBox:
        return self._move_player_box_vertically(player_box, -1)

    def _move_player_box_vertically(self, player_box: BoundingBox, direction: int) -> BoundingBox:
        reference_top_left = transform_coordinates([(player_box.x1, player_box.y1)], self.inverse_homography_matrix)[0]
        reference_bottom_right = transform_coordinates([(player_box.x2, player_box.y2)], self.inverse_homography_matrix)[0]
        dy = direction * player_box.width_height_ratio * REFERENCE_COURT_PIXEL_TO_METER_RATIO
        reference_top_left = (reference_top_left[0], round(reference_top_left[1] + dy))
        reference_bottom_right = (reference_bottom_right[0], round(reference_bottom_right[1] + dy))
        x1, y1 = transform_coordinates([reference_top_left], self.homography_matrix)[0]
        x2, y2 = transform_coordinates([reference_bottom_right], self.homography_matrix)[0]
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
        top_base_line_width = self.court_keypoints[1][0] - self.court_keypoints[0][0]
        bottom_base_line_width = self.court_keypoints[3][0] - self.court_keypoints[2][0]
        base_line_diff = bottom_base_line_width - top_base_line_width
        court_height = self.court_keypoints[2][1] - self.court_keypoints[0][1]
        player_to_top_base_line = player_box.y2 - self.court_keypoints[0][1]
        player_line_width = (player_to_top_base_line / court_height) * base_line_diff + top_base_line_width
        player_width_meters = (player_box.width / player_line_width) * DOUBLES_LINE_WIDTH
        player_height_meters = round((player_box.height / player_box.width) * player_width_meters, 2)
        print(f'Player height: {player_height_meters} m')
        return player_height_meters

    def _compute_vertical_flight_path_hit(self, frame_count: int, fps: float, initial_height: float) -> Tuple[List[float], List[float]]:
        time_seconds = frame_count / fps
        initial_vertical_velocity = round(get_initial_vertical_velocity_hit(initial_height, time_seconds), 2)
        vertical_distances, vertical_velocities = get_vertical_distances_and_velocities(
            initial_vertical_velocity, 
            initial_height, 
            time_seconds / frame_count)
        
        return vertical_distances, vertical_velocities

    def _compute_vertical_flight_path_bounce(self, frame_count: int, fps: float, initial_height: float, final_height: float) -> Tuple[List[float], List[float]]:
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
                                vertical_velocities: List[float]) -> None:
        balls: List[ReferenceBall] = []
        for j, height in enumerate(ball_heights):
            frame_number = start_frame_number + j
            ball = self.reference_frames[frame_number].ball
            balls.append(ball)
            ball.height_meters = height
            ball.vertical_velocity = vertical_velocities[j]

    def load_reference_frames(self) -> List[ReferenceFrame]:
        """
        Load reference frames from the CSV file.
        
        Returns:
            List of reference frames
        """
        df = pd.read_csv(self.reference_frames_path)
        reference_frames = []
        
        for _, row in df.iterrows():
            near_player = ReferencePlayer(
                original_bounding_box=BoundingBox(*eval(row['near_player_original_box'])),
                reference_coordinate=eval(row['near_player_reference_point']),
                action=row['near_player_action'],
            )
            far_player = ReferencePlayer(
                original_bounding_box=BoundingBox(*eval(row['far_player_original_box'])),
                reference_coordinate=eval(row['far_player_reference_point']),
                action=row['far_player_action'],
            )
            ball = ReferenceBall(
                original_bounding_box=BoundingBox(*eval(row['ball_original_box'])),
                reference_coordinate=eval(row['ball_reference_point']),
                height_meters=row['ball_height'],
                horizontal_velocity=row['ball_horizontal_velocity'],
                vertical_velocity=row['ball_vertical_velocity'],
                action=row['ball_action']
            )
            frame_number = int(row['frame_number'])
            reference_frame = ReferenceFrame(frame_number, near_player, far_player, ball)
            reference_frames.append(reference_frame)
        
        return reference_frames

    def draw(self, input_frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Draw the reference court, players, and ball on the input frames.
        
        Args:
            input_frames: List of input frames
            
        Returns:
            List of output frames with visualization
        """
        reference_frames = self.load_reference_frames()

        output_frames = []
        
        for frame_number, frame in enumerate(input_frames):
            # Draw reference court canvas and court lines
            frame = self._draw_canvas(frame)
            frame = self._draw_court(frame)
            
            # Draw players and ball
            self._draw_player_coordinates(frame, reference_frames[frame_number])
            self._draw_ball_coordinates(frame, reference_frames[frame_number].ball)    # Ball
            
            output_frames.append(frame)
            
        return output_frames

    def _draw_player_coordinates(self, frame: np.ndarray, reference_frame: ReferenceFrame) -> None:
        frame_width = frame.shape[1]
        canvas_x1 = frame_width - REFERENCE_COURT_CANVAS_WIDTH - REFERENCE_COURT_MARGIN_X
        canvas_y1 = REFERENCE_COURT_MARGIN_Y
        canvas_y2 = REFERENCE_COURT_MARGIN_Y + REFERENCE_COURT_CANVAS_DEPTH
        color_red = (0, 0, 255)
        color_blue = (255, 0, 0)
        x = int(reference_frame.near_player.reference_coordinate[0] + canvas_x1)
        y = int(reference_frame.near_player.reference_coordinate[1] + canvas_y1)
        cv2.circle(frame, (x, y), 10, color_red, -1)
        cv2.putText(frame, f"{NEAR_PLAYER_NAME}", (canvas_x1 + 10, canvas_y2 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_red, 2, cv2.LINE_AA)

        x = int(reference_frame.far_player.reference_coordinate[0] + canvas_x1)
        y = int(reference_frame.far_player.reference_coordinate[1] + canvas_y1)
        cv2.circle(frame, (x, y), 10, color_blue, -1)
        cv2.putText(frame, f"{FAR_PLAYER_NAME}", (canvas_x1 + 10, canvas_y1 + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_blue, 2, cv2.LINE_AA)

    def _draw_ball_coordinates(self, frame: np.ndarray, ball: ReferenceBall) -> None:
        frame_width = frame.shape[1]
        canvas_x1 = frame_width - REFERENCE_COURT_CANVAS_WIDTH - REFERENCE_COURT_MARGIN_X
        canvas_y1 = REFERENCE_COURT_MARGIN_Y
        color_green = (0, 255, 0)
        color_black = (0, 0, 0)
        x = int(ball.reference_coordinate[0] + canvas_x1)
        y = int(ball.reference_coordinate[1] + canvas_y1)
        cv2.circle(frame, (x, y), 5, color_green, -1)

        cv2.putText(
            frame, f"h: {ball.height_meters}", (x + 10, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_black, 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"v: {get_hypotenuse(ball.horizontal_velocity, ball.vertical_velocity)}", (x + 10, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_black, 1, cv2.LINE_AA
        )

    def _draw_court(self, frame: np.ndarray) -> np.ndarray:
        frame_width = frame.shape[1]
        canvas_x1 = frame_width - REFERENCE_COURT_CANVAS_WIDTH - REFERENCE_COURT_MARGIN_X
        canvas_y1 = REFERENCE_COURT_MARGIN_Y
        canvas_y2 = REFERENCE_COURT_MARGIN_Y + REFERENCE_COURT_CANVAS_DEPTH
        # Draw lines
        for line in REFERENCE_COURT_LINES:
            x1 = int(REFERENCE_KEYPOINTS[line[0]][0] + canvas_x1)
            y1 = int(REFERENCE_KEYPOINTS[line[0]][1] + canvas_y1)
            x2 = int(REFERENCE_KEYPOINTS[line[1]][0] + canvas_x1)
            y2 = int(REFERENCE_KEYPOINTS[line[1]][1] + canvas_y1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

        # Draw center line
        center_x = int((REFERENCE_KEYPOINTS[0][0] + REFERENCE_KEYPOINTS[1][0]) / 2 + canvas_x1)
        
        # Top center line
        top_y1 = int(REFERENCE_KEYPOINTS[0][1] + canvas_y1)
        top_y2 = int(top_y1 + CENTER_LINE_DEPTH * REFERENCE_COURT_PIXEL_TO_METER_RATIO)
        cv2.line(frame, (center_x, top_y1), (center_x, top_y2), (0, 0, 0), 2)
        
        # Bottom center line
        bottom_y1 = int(REFERENCE_KEYPOINTS[2][1] + canvas_y1)
        bottom_y2 = int(bottom_y1 - CENTER_LINE_DEPTH * REFERENCE_COURT_PIXEL_TO_METER_RATIO)
        cv2.line(frame, (center_x, bottom_y1), (center_x, bottom_y2), (0, 0, 0), 2)

        # Draw net
        net_x1 = int(REFERENCE_COURT_X1 + canvas_x1)
        net_x2 = int(REFERENCE_COURT_X2 + canvas_x1)
        net_y = int((canvas_y1 + canvas_y2) / 2)
        cv2.line(frame, (net_x1, net_y), (net_x2, net_y), (255, 0, 0), 2)

        # Draw key points
        for keypoint in REFERENCE_KEYPOINTS:
            x = int(keypoint[0] + canvas_x1)
            y = int(keypoint[1] + canvas_y1)
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

        return frame

    def _draw_canvas(self, frame: np.ndarray) -> np.ndarray:
        frame_width = frame.shape[1]
        canvas_x1 = frame_width - REFERENCE_COURT_MARGIN_X - REFERENCE_COURT_CANVAS_WIDTH
        canvas_x2 = frame_width - REFERENCE_COURT_MARGIN_X
        canvas_y1 = REFERENCE_COURT_MARGIN_Y
        canvas_y2 = REFERENCE_COURT_MARGIN_Y + REFERENCE_COURT_CANVAS_DEPTH
        # Create a blank image with the same size as the input frame
        shapes = np.zeros_like(frame, np.uint8)
        
        # Draw a filled white rectangle for the canvas
        cv2.rectangle(
            shapes, 
            (int(canvas_x1), int(canvas_y1)), 
            (int(canvas_x2), int(canvas_y2)), 
            (255, 255, 255), 
            cv2.FILLED
        )
        
        output_frame = frame.copy()
        
        # Apply transparent overlay
        alpha = 0.5
        mask = shapes.astype(bool)
        output_frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return output_frame
