from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import torch
from .utils import get_bounding_box_center_point, get_distance_between_point_and_line, get_distance_between_points
import numpy as np

# This class is used to track the ball in a video using YOLOv8 model
# It uses the YOLOv8 model to detect the ball in each frame of the video
#
# ball_position is a key-value pair with track ID as key and bounding box as value
# ball_positions_per_frame is a dictionary with track IDs as keys and bounding boxes as values
# ball_positions_all_frames is a list of dictionaries with track IDs as keys and bounding boxes as values
#
# Drag force on a tennis ball: Fd = 1/2*Cd*p*A*v^2
# Cd: drag coefficient: 0.55 - 0.65
# p: air density: 1.225 kg/m3 at see level
# A: cross-sectional area of tennis ball: 3.14*(0.066/2)^2 = 0.0034 m2
# v: velocity of the tennis ball
# Which means on average: Fd = 1/2*0.6*1.225*0.0034*v^2 => Fd = 0.00125*v^2
# A 30 m/s ball will have drag force: 0.00125*30^2 = 1.125 N, 
# Compare to gravitational force: m*g = 0.057*9.8 = 0.559 N
# 
# Condition             Velocity    Time to Bounce  Travel Distance Launching angle
# No air no spin        30 m/s      0.452 s         13.56 m         
# Air no spin           30 m/s      0.470 s         12.46 m         4.0 - 8.1
# Air top spin (20 rps) 30 m/s      0.410 s         10.85 m         5.5 - 11.9
#
# Magnus force on a tennis ball: Fl = Cl*p*A*v^2
# Cl: lift coefficiency: Cl = 1/(2+V/Vspin)
#     where Vspin = R*w
#     R: radius of the ball: 0.033 m
#     w: angular speed 100 - 500 radians/sec, translates to 16 - 80 revs/sec
# A 30 m/s ball with 50 revs/sec: Cl = 1/(2+30/(0.033*300)) = 0.20
# Fl = 0.2*1.225*0.0034*30^2 = 0.75 N
#
# The combined drag and lift coefficiency: Cdl = Cd+1/(22.5+4.2(V/Vspin)^2.5)^0.4
# If Cd is 0.6, V is 30m and Vspin is 9.9m/s, Cdl = 0.6+1/(22.5+4.2(30/9.9)^2.5)^0.4 = 0.77
#
# Vertical terminal velocity Vt of the tennis ball with drag only can be calculated using
# weight = drag 
# => 
# mg = Cd*p*A*Vt^2/2 => Vt^2 = (2*m*g)/(Cd*p*A)
# => 
# 0.057*9.8*2 = 0.6*1.225*0.0034*Vt^2
# =>
# Vt = (1.1172/0.002499)^0.5 = 477.0588^0.5 = 21.84
# 
# Vertical acceleration changes with time and velocity, for vertical ascent
# ma = -mg - Cd*p*A*v^2/2
# =>
# a = -g - Cd*p*A*v^2/(2*m)
# => use Vt^2 = (2*m*g)/(Cd*p*A)
# a = -g*(1+v^2/Vt^2)
#
# Vertical velocity in terms of time
# a = dv/dt = -g*(1+v^2/Vt^2)
# => using integration, limits and trigonometric identiy
# V/Vt = (V0-Vt-tan(g*t/Vt))/(Vt+V0*tan(g*t/Vt))
# 
# Vertical location in terms of vertical velocity
# y = (Vt^2/2*g)*ln((V0^2+Vt^2)/(V^2+Vt^2))
# y(max) = (Vt^2/2*g)*ln((V0^2+Vt^2)/(Vt^2))
#
# Horizontal location in terms of time
# ma = -Cd*p*A*u^2/2
# =>
# a = -Cd*p*A*u^2/(2*m)
# => use Vt^2 = (2*m*g)/(Cd*p*A)
# a = du/dt = -g*u^2/Vt^2
# => integrating and limit
# x = (Vt^2/g)*ln((Vt^2_g*U0*t)/Vt^2)
#
class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available else 'cpu')
        self.model.to(device)

    # Detect ball positions in a list of frames
    def dectect_ball_positions_all_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_positions_all_frames = []
        if read_from_stub and stub_path is not None:
            # Read frames from stub
            with open(stub_path, 'rb') as f:
                ball_positions_all_frames = pickle.load(f)
            return ball_positions_all_frames
            
        for frame in frames:
            ball_positions_all_frames.append(self.detect_ball_positions_per_frame(frame))
        self.remove_all_extra_balls_detected(ball_positions_all_frames)

        if stub_path is not None:
            # Save frames to stub
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_positions_all_frames, f)
        return ball_positions_all_frames

    # Detect ball positions in a single frame
    # Returns a dictionary with track IDs as keys and bounding boxes as values
    # Bounding box format: [x1, y1, x2, y2]
    # x1, y1 are the top-left coordinates and x2, y2 are the bottom-right coordinates
    def detect_ball_positions_per_frame(self, frame):
        # Perform prediction on the frame
        # The model will return a list of results, we take the first one
        # The 'conf' parameter is set to 0.15 to filter out low-confidence detections
        results = self.model.predict(frame, conf=0.15)[0]
        ball_positions_per_frame = {}
        track_id = 1
        for box in results.boxes:
            # print(box)
            # The bounding boxes are stored in the 'xyxy' attribute of the boxes
            # The 'xyxy' attribute contains a list of lists, where each inner list
            # contains the coordinates of the bounding box in the format [x1, y1, x2, y2]
            bounding_box = box.xyxy.tolist()[0]
            # We only track 1 ball here, so the track ID is set to 1
            ball_positions_per_frame[track_id] = bounding_box
            track_id += 1
        return ball_positions_per_frame

    def remove_all_extra_balls_detected(self, ball_positions_all_frames):
        for i, ball_positions in enumerate(ball_positions_all_frames):
            if len(ball_positions) < 2:
                continue
            last_ball_position = self.find_last_ball_position(ball_positions_all_frames, i)
            if last_ball_position is None:
                ball_positions_all_frames[i] = {1:ball_positions[1]}
                continue
            next_ball_position = self.find_next_ball_position(ball_positions_all_frames, i)
            if next_ball_position is None:
                ball_positions_all_frames[i] = {1:ball_positions[1]}
                continue
            ball_positions_all_frames[i] = {1:self.find_best_ball_position(last_ball_position, ball_positions, next_ball_position)}

    def find_last_ball_position(self, ball_positions_all_frames, index):
        if index == 0:
            return None
        for i in range(index - 1, 0, -1):
            if len(ball_positions_all_frames[i]) == 1:
                return ball_positions_all_frames[i][1]
        return None

    def find_next_ball_position(self, ball_positions_all_frames, index):
        if index == len(ball_positions_all_frames) - 1:
            return None
        for i in range(index + 1, len(ball_positions_all_frames)):
            if len(ball_positions_all_frames[i]) == 1:
                return ball_positions_all_frames[i][1]
        return None

    def find_best_ball_position(self, last_ball_position, ball_positions, next_ball_position):
        line_start = get_bounding_box_center_point(last_ball_position)
        line_end = get_bounding_box_center_point(next_ball_position)
        min_distance = np.inf
        best_ball_id = 0
        for id, ball_position in ball_positions.items():
            point = get_bounding_box_center_point(ball_position)
            distance = get_distance_between_point_and_line(point, line_start, line_end)
            if distance < min_distance:
                min_distance = distance
                best_ball_id = id
        return ball_positions[best_ball_id]

    def interpolate_ball_positions(self, ball_positions_all_frames):
        # print('Interpolating ball positions: ', ball_positions_all_frames)
        # The list comprehension extracts the bounding boxes from the dictionaries
        # and creates a list of lists, if the track ID '1' is missing in the dictionary,
        # it returns an empty list
        bounding_boxes_all_frames = [x.get(1,[]) for x in ball_positions_all_frames]
        # Convert the list of lists to a DataFrame
        # The DataFrame will have columns ['x1', 'y1', 'x2', 'y2']
        bounding_boxes_data_frame = pd.DataFrame(bounding_boxes_all_frames, columns=['x1','y1','x2','y2'])

        bounding_boxes_data_frame['width'] = bounding_boxes_data_frame['x2'] - bounding_boxes_data_frame['x1']
        bounding_boxes_data_frame['height'] = bounding_boxes_data_frame['y2'] - bounding_boxes_data_frame['y1']
        average_width = bounding_boxes_data_frame['width'].mean()
        print(f"Average Width: {average_width}")
        average_height = bounding_boxes_data_frame['height'].mean()
        print(f"Average Height: {average_height}")
        bounding_boxes_data_frame.drop(columns=['width', 'height'], inplace=True)

        # Interpolate the missing values in the DataFrame
        # The 'interpolate' method will fill the missing values using linear interpolation
        bounding_boxes_data_frame = bounding_boxes_data_frame.interpolate()
        # The 'bfill' method will fill the missing values using backward fill
        bounding_boxes_data_frame = bounding_boxes_data_frame.bfill()
        # The 'ffill' method will fill the missing values using forward fill
        bounding_boxes_data_frame = bounding_boxes_data_frame.ffill()

        # Convert the DataFrame back to a NumPy array, then to a list of lists
        # The list comprehension creates a list of dictionaries
        # where each dictionary has track ID '1' as key and the bounding box as value
        ball_positions_all_frames = [{1:x} for x in bounding_boxes_data_frame.to_numpy().tolist()]

        # print('Interpolated ball positions: ', ball_positions_all_frames)
        return ball_positions_all_frames
    
    def custom_interpolate_ball_positions(self, ball_positions_all_frames):
        bounding_boxes_all_frames = [x.get(1,[]) for x in ball_positions_all_frames]

        average_bounding_box_size = self.get_average_bounding_box_size(bounding_boxes_all_frames)
        missing_ranges = self.find_missing_ranges(bounding_boxes_all_frames)
        for missing_ranges in missing_ranges:
            self.interpolate_missing_range(missing_ranges, bounding_boxes_all_frames, average_bounding_box_size)

        bounding_boxes_data_frame = pd.DataFrame(bounding_boxes_all_frames, columns=['x1','y1','x2','y2'])
        bounding_boxes_data_frame = bounding_boxes_data_frame.bfill()
        bounding_boxes_data_frame = bounding_boxes_data_frame.ffill()
        ball_positions_all_frames = [{1:x} for x in bounding_boxes_data_frame.to_numpy().tolist()]
        return ball_positions_all_frames

    def get_average_bounding_box_size(self, bounding_boxes_all_frames):
        bounding_boxes_data_frame = pd.DataFrame(bounding_boxes_all_frames, columns=['x1','y1','x2','y2'])

        bounding_boxes_data_frame['width'] = bounding_boxes_data_frame['x2'] - bounding_boxes_data_frame['x1']
        bounding_boxes_data_frame['height'] = bounding_boxes_data_frame['y2'] - bounding_boxes_data_frame['y1']
        average_width = bounding_boxes_data_frame['width'].mean()
        average_height = bounding_boxes_data_frame['height'].mean()
        return (average_width, average_height)

    def find_missing_ranges(self, bounding_boxes_all_frames):
        missing_ranges = []
        start_index = None
        for i, bounding_box in enumerate(bounding_boxes_all_frames):
            if len(bounding_box) == 0:
                if start_index is None:
                    start_index = i
            else:
                if start_index is not None:
                    missing_ranges.append((start_index, i - 1))
                    start_index = None
        if len(missing_ranges) > 0 and missing_ranges[0][0] == 0:
            missing_ranges = missing_ranges[1:]
        return missing_ranges

    def interpolate_missing_range(self, missing_range, bounding_boxes_all_frames, average_bounding_box_size):
        start_bounding_box = bounding_boxes_all_frames[missing_range[0] - 1]
        end_bounding_box = bounding_boxes_all_frames[missing_range[1] + 1]
        start_position = get_bounding_box_center_point(start_bounding_box)
        end_position = get_bounding_box_center_point(end_bounding_box)
        steps = missing_range[1] - missing_range[0] + 3
        sequence_x = self.exponential_deceleration(start_position[0], end_position[0], steps)
        sequence_x = sequence_x[1:-1]
        sequence_y = self.exponential_deceleration(start_position[1], end_position[1], steps)
        sequence_y = sequence_y[1:-1]
        for i in range(len(sequence_x)):
            bounding_box = [
                sequence_x[i] - average_bounding_box_size[0] / 2, 
                sequence_y[i] - average_bounding_box_size[1] / 2, 
                sequence_x[i] + average_bounding_box_size[0] / 2, 
                sequence_y[i] + average_bounding_box_size[1] / 2]
            bounding_boxes_all_frames[missing_range[0] + i] = bounding_box

    def exponential_deceleration(self, start, end, steps, exponent=1.5):
        if steps <= 0:
            return []
        if steps == 1:
            return [start]
        sequence = [start + (end - start) * (1 - (1 - i / (steps - 1)) ** exponent) for i in range(steps)]
        return sequence

    def get_ball_shot_frame_indexes(self, ball_positions_all_frames):
        bounding_boxes_all_frames = [x.get(1, []) for x in ball_positions_all_frames]
        # convert the list into pandas dataframe
        bounding_boxes = pd.DataFrame(bounding_boxes_all_frames, columns=['x1','y1','x2','y2'])

        bounding_boxes['mid_y'] = (bounding_boxes['y1'] + bounding_boxes['y2']) / 2
        bounding_boxes['mid_y_rolling_mean'] = bounding_boxes['mid_y'].rolling(window = 5, min_periods = 1, center = False).mean()
        bounding_boxes['delta_y'] = bounding_boxes['mid_y_rolling_mean'].diff()
        bounding_boxes['ball_hit'] = 0

        # print(bounding_boxes)

        minimum_change_frames_for_hit = 25
        for i in range(1, len(bounding_boxes) - int(minimum_change_frames_for_hit * 1.2)):
            negative_change = bounding_boxes.at[i, 'delta_y'] > 0 and bounding_boxes.at[i+1, 'delta_y'] < 0
            positive_change = bounding_boxes.at[i, 'delta_y'] < 0 and bounding_boxes.at[i+1, 'delta_y'] > 0

            if negative_change or positive_change:
                change_count = 0
                # Count how many following frames have the same change
                # In this case, if 25 out of 30 frames have the same change, we consider it a hit 
                for j in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    still_negative = bounding_boxes.at[i, 'delta_y'] > 0 and bounding_boxes.at[j, 'delta_y'] < 0
                    still_positive = bounding_boxes.at[i, 'delta_y'] < 0 and bounding_boxes.at[j, 'delta_y'] > 0

                    if negative_change and still_negative:
                        change_count += 1
                    elif positive_change and still_positive:
                        change_count += 1
            
                if change_count >= minimum_change_frames_for_hit:
                    bounding_boxes.loc[i, 'ball_hit'] = 1

        return bounding_boxes[bounding_boxes['ball_hit'] == 1].index.tolist()

    def get_ball_hits(self, ball_positions_all_frames):
        bounding_boxes = [x.get(1, []) for x in ball_positions_all_frames]
        centre_points = []
        for bounding_box in bounding_boxes:
            centre_points.append(get_bounding_box_center_point(bounding_box))
        delta_ys = []
        delta_ys.append(0)
        for i in range(1, len(centre_points)):
            delta_ys.append(centre_points[i][1] - centre_points[i-1][1])
        velocities = []
        velocities.append(0)
        for i in range(1, len(centre_points)):
            velocity = get_distance_between_points(centre_points[i], centre_points[i - 1])
            if delta_ys[i] < 0:
                velocity *= -1
            velocities.append(int(velocity))
        accelerations = []
        accelerations.append(0)
        accelerations.append(0)
        for i in range(2, len(velocities)):
            accelerations.append(int(velocities[i] - velocities[i - 1]))
        df = pd.DataFrame({
            'bounding_box': bounding_boxes,
            'center_point': centre_points,
            'delta_y': delta_ys,
            'velocity': velocities,
            'acceleration': accelerations
        })
        df['frame'] = df.index
        df['velocity_rolling_mean'] = df['velocity'].rolling(window = 5, min_periods = 1, center = False).mean().round().astype(int)
        df['delta_velocity'] = df['velocity_rolling_mean'].diff().round()

        df.to_csv('ball_trajectory_data.csv', index=False)

    # Draw bounding boxes on the frames
    def draw(self, input_frames, ball_positions_all_frames):
        output_frames = []
        for frame, ball_positions_per_frame in zip(input_frames, ball_positions_all_frames):
            for track_id, bounding_box in ball_positions_per_frame.items():
                # Tuple unpacking of the bounding box
                x1, y1, x2, y2 = bounding_box
                # Convert the bounding box coordinates to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Draw the bounding box on the frame
                # The bounding box is drawn in red color with a thickness of 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Draw the track ID above the bounding box
                # The track ID is drawn in red color with a thickness of 2
                # The text is drawn 10 pixels above the bounding box
                # The font scale is set to 1
                cv2.putText(frame, str(track_id), (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            output_frames.append(frame)
        return output_frames
