import cv2
import numpy as np
from scipy.spatial import distance
from .utils import get_bottom_line_center_point, get_bounding_box_center_point, get_distance_between_points, \
    get_initial_horizontal_velocity, get_distance_by_time
from .constants import DOUBLES_LINE_WIDTH, DOUBLES_ALLEY_WIDTH, RUN_BACK_DEPTH, \
    SIDE_RUN_WIDTH, HALF_COURT_DEPTH, NO_MANS_LAND_DEPTH, REFERENCE_COURT_MARGIN_X, \
    REFERENCE_COURT_MARGIN_Y, CENTER_LINE_DEPTH

# The reference court is a scaled version of a standard tennis court.
# The court in our image is most likely an isosceles trapezium, but our 
# reference court is a rectangle.
# The only parameter we need to create our reference court is the width 
# of the court, we can work out the rest of the court dimensions.
#
# Here are the 14 key points of the reference court:
#
# 0-4---------6-1
# | |         | |
# | |         | |
# | 8----C----9 |
# | |    |    | |
# | |    |    | |
# +-+----+----+-+
# | |    |    | |
# | |    |    | |
# | A----D----B |
# | |         | |
# | |         | |
# 2-5---------7-3
#                  
class ReferenceCourt():
    def __init__(self, frame_width, frame_height, canvas_width):
        self.canvas_width = canvas_width
        self.pixel_to_meter_ratio = self.canvas_width / (DOUBLES_LINE_WIDTH + 2 * SIDE_RUN_WIDTH)
        self.canvas_depth = ((HALF_COURT_DEPTH + RUN_BACK_DEPTH) * 2) * self.pixel_to_meter_ratio

        # Initialize canvas positions on the frame
        self.canvas_x2 = frame_width - REFERENCE_COURT_MARGIN_X
        self.canvas_x1 = self.canvas_x2 - self.canvas_width
        self.canvas_y2 = frame_height - REFERENCE_COURT_MARGIN_Y
        self.canvas_y1 = self.canvas_y2 - self.canvas_depth

        # Initialize court positions on the canvas
        self.court_x1 = SIDE_RUN_WIDTH * self.pixel_to_meter_ratio
        self.court_x2 = self.canvas_width - self.court_x1
        self.court_y1 = RUN_BACK_DEPTH * self.pixel_to_meter_ratio
        self.court_y2 = self.canvas_depth - self.court_y1

        self.doubles_alley_width = DOUBLES_ALLEY_WIDTH * self.pixel_to_meter_ratio
        self.no_mans_land_depth = NO_MANS_LAND_DEPTH * self.pixel_to_meter_ratio

        # Initialize 14 key points
        self.keypoints = [
            (self.court_x1,                              self.court_y1), # 0
            (self.court_x2,                              self.court_y1), # 1
            (self.court_x1,                              self.court_y2), # 2
            (self.court_x2,                              self.court_y2), # 3
            (self.court_x1 + self.doubles_alley_width,   self.court_y1), # 4
            (self.court_x1 + self.doubles_alley_width,   self.court_y2), # 5
            (self.court_x2 - self.doubles_alley_width,   self.court_y1), # 6
            (self.court_x2 - self.doubles_alley_width,   self.court_y2), # 7
            (self.court_x1 + self.doubles_alley_width,   self.court_y1 + self.no_mans_land_depth), # 8
            (self.court_x2 - self.doubles_alley_width,   self.court_y1 + self.no_mans_land_depth), # 9
            (self.court_x1 + self.doubles_alley_width,   self.court_y2 - self.no_mans_land_depth), # 10
            (self.court_x2 - self.doubles_alley_width,   self.court_y2 - self.no_mans_land_depth), # 11
            ((self.court_x1 + self.court_x2) / 2,        self.court_y1 + self.no_mans_land_depth), # 12
            ((self.court_x1 + self.court_x2) / 2,        self.court_y2 - self.no_mans_land_depth)  # 13
        ]

        # Initialize court lines, 4 vertical and 4 horizontal lines
        self.court_lines = [(0, 2), (4, 5), (6, 7), (1, 3), (0, 1), (8, 9), (12, 13), (10, 11), (2, 3)]

        self.homography_points = [
            [0, 1, 2, 3],    # doubles court
            [0, 6, 2, 7],    # right doubles alley plus singles court
            [4, 1, 5, 3],    # left doubles alley plus singles court
            [4, 6, 5, 7],    # singles court
            [0, 4, 2, 5],    # right doubles alley
            [6, 1, 7, 3],    # left doubles alley
            [4, 6, 10, 11],  # top no mans land plus mini court
            [10, 11, 5, 7]   # bottom no mans land
        ]

    def homograph_keypoints(self, given_keypoints):
        self.homography_matrix = None
        distance_max = np.inf
        reference_keypoints = np.array(self.keypoints, dtype=np.float32).reshape((-1, 1, 2))
        for homography_point in self.homography_points:
            reference_configuration = [
                self.keypoints[homography_point[0]], 
                self.keypoints[homography_point[1]], 
                self.keypoints[homography_point[2]], 
                self.keypoints[homography_point[3]]
            ]
            given_configuration = [
                given_keypoints[homography_point[0]], 
                given_keypoints[homography_point[1]], 
                given_keypoints[homography_point[2]], 
                given_keypoints[homography_point[3]]
            ]
            if not any([None in given_configuration]):
                matrix, _ = cv2.findHomography(np.float32(reference_configuration), 
                                            np.float32(given_configuration), 
                                            method=0)
                transformed_keypoints = cv2.perspectiveTransform(reference_keypoints, matrix)
                distances = []
                for i in range(12):
                    if i not in homography_point and given_keypoints[i] is not None:
                        distances.append(distance.euclidean(given_keypoints[i], transformed_keypoints[i][0]))
                distance_median = np.mean(distances)
                if distance_median < distance_max:
                    self.homography_matrix = matrix
                    distance_max = distance_median
        self.inverse_homography_matrix = np.linalg.inv(self.homography_matrix)
        # best_keypoints is a list of list [[x1 y1], [x2 y2], ...]
        best_keypoints = cv2.perspectiveTransform(reference_keypoints, self.homography_matrix)
        # turn best_keypoints into a list of tuples [(x1, y1), (x2, y2), ...]
        self.homographied_keypoints = [tuple(map(int, point)) for point in best_keypoints.reshape(-1, 2)]
        # We preserve the refined keypoints from the given keypoints which are better than the
        # predicted keypoints. We only use the keypoints which refined keypoints can work out.
        for i in range(len(self.homographied_keypoints)):
            if given_keypoints[i] is not None:
                self.homographied_keypoints[i] = given_keypoints[i]

    def convert_player_coordinates(self, player_positions):
        self.player_coordinates = self.convert_to_reference_coordinates(player_positions)

    def convert_to_reference_coordinates(self, positions_all_frames):
        reference_coordinates = []
        for frame_number, positions_per_frame in enumerate(positions_all_frames):
            reference_coordinates_per_frame = {}
            for track_id, bounding_box in positions_per_frame.items():
                original_coordinate = get_bottom_line_center_point(bounding_box)
                reference_coordinate = self.get_reference_coordinate(original_coordinate)
                reference_coordinates_per_frame[track_id] = reference_coordinate
            reference_coordinates.append(reference_coordinates_per_frame)
        return reference_coordinates

    def get_reference_coordinate(self, original_coordinate):
        original_coordinate = np.array(original_coordinate, dtype=np.float32).reshape((-1, 1, 2))
        reference_coordinate = cv2.perspectiveTransform(original_coordinate, self.inverse_homography_matrix)
        reference_coordinate = reference_coordinate[0][0]
        return int(reference_coordinate[0]), int(reference_coordinate[1])

    def get_original_coordinate(self, reference_coordinate):
        reference_coordinate = np.array(reference_coordinate, dtype=np.float32).reshape((-1, 1, 2))
        original_coordinate = cv2.perspectiveTransform(reference_coordinate, self.homography_matrix)
        original_coordinate = original_coordinate[0][0]
        return int(original_coordinate[0]), int(original_coordinate[1])

    # Convert detected ball coordinates to reference coordinates with calculated flight path
    def convert_ball_coordinates(self, player_positions, near_player, far_player, ball_positions, hits_and_bounces, fps):
        self.ball_coordinates = self.convert_to_reference_coordinates(ball_positions)
        i = 0
        while i < len(hits_and_bounces) - 1:
            if hits_and_bounces[i] is None or hits_and_bounces[i + 1] is None:
                i += 1
                continue
            ball_point_1 = get_bounding_box_center_point(ball_positions[hits_and_bounces[i]][1])
            ball_point_2 = get_bounding_box_center_point(ball_positions[hits_and_bounces[i + 1]][1])
            frame_number = hits_and_bounces[i + 1] - hits_and_bounces[i]
            player_box = self.get_player_box(player_positions, hits_and_bounces, i, far_player, near_player)
            start_point, end_point = self.get_start_and_end_points(player_box, i, ball_point_1, ball_point_2)
            calculated_coordinates = self.calculate_flight_path(start_point, end_point, frame_number, fps)
            self.replace_ball_coordinates(hits_and_bounces, i, calculated_coordinates)
            i += 1

    # Replace detected ball coordinates with calculated coordinates so that we 
    # can take the ball height out of the equation
    def replace_ball_coordinates(self, hits_and_bounces, hits_and_bounces_index, calculated_coordinates):
        for j in range(len(calculated_coordinates)):
            frame_number = hits_and_bounces[hits_and_bounces_index] + j
            self.ball_coordinates[frame_number] = {1: calculated_coordinates[j]}

    # Get the start and end points of a bounce-hit or hit-bounce, assume the 
    # hit point is at the players feet
    def get_start_and_end_points(self, player_box, hits_and_bounces_index, ball_point_1, ball_point_2):
        if hits_and_bounces_index % 2 == 0:
            # Hit and bounce
            hit_point = (ball_point_1[0], player_box[3])
            start_point = self.get_reference_coordinate(hit_point)
            end_point = self.get_reference_coordinate(ball_point_2)
        else:
            # Bounce and hit
            hit_point = (ball_point_2[0], player_box[3])
            start_point = self.get_reference_coordinate(ball_point_1)
            end_point = self.get_reference_coordinate(hit_point)
        return start_point, end_point

    # Get the player bounding box for the hitting player
    def get_player_box(self, player_positions, hits_and_bounces, hits_and_bounces_index, far_player, near_player):
        if hits_and_bounces_index % 4 == 0:
            # Far hit and bounce
            player_box = [int(x) for x in player_positions[hits_and_bounces[hits_and_bounces_index]][far_player]]
        elif hits_and_bounces_index % 4 == 1:
            # Near bounce and hit
            player_box = [int(x) for x in player_positions[hits_and_bounces[hits_and_bounces_index + 1]][near_player]]
        elif hits_and_bounces_index % 4 == 2:
            # Near hit and bounce
            player_box = [int(x) for x in player_positions[hits_and_bounces[hits_and_bounces_index]][near_player]]
        elif hits_and_bounces_index % 4 == 3:
            # Far bounce and hit
            player_box = [int(x) for x in player_positions[hits_and_bounces[hits_and_bounces_index + 1]][far_player]]
        return player_box

    # Calculatet the balls flight path using trajectory equation with air drag
    def calculate_flight_path(self, start_point, end_point, frame_number, fps):
        print(f'calculate_hit_bounce: start point: {start_point}, end point: {end_point}, frame number: {frame_number}')
        x = get_distance_between_points(start_point, end_point) / self.pixel_to_meter_ratio
        t = frame_number /fps
        initial_velocity = get_initial_horizontal_velocity(x, t)
        print(f'x: {x}, t: {t}, initial velocity: {initial_velocity}')
        calculated_coordinates = []
        calculated_coordinates.append(start_point)
        # The last frame will be calculated by next calculate flight path
        for i in range(1, frame_number):
            ti = i / fps
            di = get_distance_by_time(initial_velocity, ti)
            ratio = di / x
            coordinate_x = start_point[0] + ratio * (end_point[0] - start_point[0])
            coordinate_y = start_point[1] + ratio * (end_point[1] - start_point[1])
            calculated_coordinates.append((int(coordinate_x), int(coordinate_y)))
        return calculated_coordinates

    def draw(self, input_frames):
        output_frames = []
        for frame_number, frame in enumerate(input_frames):
            frame = self.draw_canvas(frame)
            frame = self.draw_court(frame)
            self.draw_coordinates(frame, frame_number, self.player_coordinates, 1, 10, color=(0, 0, 255))
            self.draw_coordinates(frame, frame_number, self.player_coordinates, 2, 10, color=(255, 0, 0))
            self.draw_coordinates(frame, frame_number, self.ball_coordinates, 1, 5, color=(0, 255, 0))
            output_frames.append(frame)
        return output_frames

    def draw_coordinates(self, frame, frame_number, coordinates, id, size, color=(0, 255, 0)):
        for track_id, coordinate in coordinates[frame_number].items():
            if track_id == id:
                x = int(coordinate[0] + self.canvas_x1)
                y = int(coordinate[1] + self.canvas_y1)
                cv2.circle(frame, (x, y), size, color, -1)

    def draw_court(self, frame):
        # Draw Lines
        for line in self.court_lines:
            x1 = int(self.keypoints[line[0]][0] + self.canvas_x1)
            y1 = int(self.keypoints[line[0]][1] + self.canvas_y1)
            x2 = int(self.keypoints[line[1]][0] + self.canvas_x1)
            y2 = int(self.keypoints[line[1]][1] + self.canvas_y1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

        # Draw center line
        center_line_x = int((self.keypoints[0][0] + self.keypoints[1][0]) / 2 + self.canvas_x1)
        center_line_y1 = int(self.keypoints[0][1] + self.canvas_y1)
        center_line_y2 = int(center_line_y1 + CENTER_LINE_DEPTH * self.pixel_to_meter_ratio)
        cv2.line(frame, (center_line_x, center_line_y1), (center_line_x, center_line_y2), (0, 0, 0), 2)
        center_line_y1 = int(self.keypoints[2][1] + self.canvas_y1)
        center_line_y2 = int(center_line_y1 - CENTER_LINE_DEPTH * self.pixel_to_meter_ratio)
        cv2.line(frame, (center_line_x, center_line_y1), (center_line_x, center_line_y2), (0, 0, 0), 2)

        # Draw net
        net_x1 = int(self.court_x1 + self.canvas_x1)
        net_x2 = int(self.court_x2 + self.canvas_x1)
        net_y = int((self.canvas_y1 + self.canvas_y2) / 2)
        cv2.line(frame, (net_x1, net_y), (net_x2, net_y), (255, 0, 0), 2)

        # Draw key points on the frame with solid red circle
        for keypoint in self.keypoints:
            x = int(keypoint[0] + self.canvas_x1)
            y = int(keypoint[1] + self.canvas_y1)
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

        return frame

    def draw_canvas(self, frame):
        # Create a blank image with the same size as the input frame
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (int(self.canvas_x1), int(self.canvas_y1)), (int(self.canvas_x2), int(self.canvas_y2)), 
                      (255, 255, 255), cv2.FILLED)
        outout_frame = frame.copy()
        # Transparent overlay
        alpha=0.5
        # Define the mask for the area where the rectangle is drawn
        mask = shapes.astype(bool)
        # Blend the original frame with the shapes image using the mask
        # so that the rectangle is transparent
        outout_frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return outout_frame

