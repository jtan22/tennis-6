from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import torch
from .utils import get_bounding_box_center_point, get_distance_between_point_and_line, get_distance_between_points
import numpy as np
import copy
from scipy.signal import find_peaks

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
# U0 = (Vt^2)*(e^(g*x/Vt^2) - 1)/g*t
#
class BallTracker:
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available else 'cpu')
        self.model.to(device)


    # Detect ball positions in a list of frames
    def dectect_ball_positions_all_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None:
            # Read frames from stub
            with open(stub_path, 'rb') as f:
                self.multiple_ball_positions = pickle.load(f)
            self.frame_count = len(self.multiple_ball_positions)
            print(f'Loaded multiple ball positions size: {self.frame_count}')

            return
            
        self.multiple_ball_positions = []
        for frame in frames:
            self.multiple_ball_positions.append(self.detect_ball_positions_per_frame(frame))
        self.frame_count = len(self.multiple_ball_positions)
        print(f'Detected multiple ball positions size: {self.frame_count}')

        if stub_path is not None:
            # Save frames to stub
            with open(stub_path, 'wb') as f:
                pickle.dump(self.multiple_ball_positions, f)
            return

    # Detect ball positions in a single frame
    # Returns a dictionary with track IDs as keys and bounding boxes as values
    # Bounding box format: [x1, y1, x2, y2]
    # x1, y1 are the top-left coordinates and x2, y2 are the bottom-right coordinates
    def detect_ball_positions_per_frame(self, frame):
        # Perform prediction on the frame
        # The model will return a list of results, we take the first one
        # The 'conf' parameter is set to 0.15 to filter out low-confidence detections
        results = self.model.predict(frame, conf=0.15)[0]
        ball_positions = {}
        track_id = 1
        for box in results.boxes:
            # print(box)
            # The bounding boxes are stored in the 'xyxy' attribute of the boxes
            # The 'xyxy' attribute contains a list of lists, where each inner list
            # contains the coordinates of the bounding box in the format [x1, y1, x2, y2]
            bounding_box = box.xyxy.tolist()[0]
            # We only track 1 ball here, so the track ID is set to 1
            ball_positions[track_id] = bounding_box
            track_id += 1
        return ball_positions

    def analyse_ball_positions_all_frames(self, fps):
        self.remove_all_extra_balls_detected()
        self.single_centre_points = self.find_centre_points(self.single_ball_positions)
        self.clean_up_ball_positions()
        self.clean_centre_points = self.find_centre_points(self.clean_ball_positions)

        self.print_data_frame()
        
        self.find_ball_hits(fps)

    # The detection may result in multiple balls, let's pick the best ball among them
    def remove_all_extra_balls_detected(self):
        self.single_ball_positions = copy.deepcopy(self.multiple_ball_positions)
        for i, ball_positions in enumerate(self.single_ball_positions):
            if len(ball_positions) < 2:
                continue
            last_ball_position = self.find_last_ball_position(i)
            if last_ball_position is None:
                self.ball_positions[i] = {1:ball_positions[1]}
                continue
            next_ball_position = self.find_next_ball_position(i)
            if next_ball_position is None:
                self.single_ball_positions[i] = {1:ball_positions[1]}
                continue
            self.single_ball_positions[i] = {1:self.find_best_ball_position(last_ball_position, ball_positions, next_ball_position)}
        print(f'Single ball positions size: {len(self.single_ball_positions)}')

    # Find the last available ball position which has only 1 ball
    def find_last_ball_position(self, index):
        if index == 0:
            return None
        for i in range(index - 1, 0, -1):
            if len(self.multiple_ball_positions[i]) == 1:
                return self.multiple_ball_positions[i][1]
        return None

    # Find the next available ball position which has only 1 ball
    def find_next_ball_position(self, index):
        if index == len(self.multiple_ball_positions) - 1:
            return None
        for i in range(index + 1, len(self.multiple_ball_positions)):
            if len(self.multiple_ball_positions[i]) == 1:
                return self.multiple_ball_positions[i][1]
        return None

    # Find the best ball position by measuring the distance between the ball position 
    # and the last and next ball positions, whichever one has the shortest distance wins
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

    def interpolate_ball_positions_old(self):
        # print('Interpolating ball positions: ', ball_positions_all_frames)
        # The list comprehension extracts the bounding boxes from the dictionaries
        # and creates a list of lists, if the track ID '1' is missing in the dictionary,
        # it returns an empty list
        bounding_boxes = [x.get(1,[]) for x in self.single_ball_positions]
        # Convert the list of lists to a DataFrame
        # The DataFrame will have columns ['x1', 'y1', 'x2', 'y2']
        df = pd.DataFrame(bounding_boxes, columns=['x1','y1','x2','y2'])

        # Interpolate the missing values in the DataFrame
        # The 'interpolate' method will fill the missing values using linear interpolation
        df = df.interpolate()
        # The 'bfill' method will fill the missing values using backward fill
        df = df.bfill()
        # The 'ffill' method will fill the missing values using forward fill
        df = df.ffill()

        # Convert the DataFrame back to a NumPy array, then to a list of lists
        # The list comprehension creates a list of dictionaries
        # where each dictionary has track ID '1' as key and the bounding box as value
        self.complete_ball_positions = [{1:x} for x in df.to_numpy().tolist()]
        print(f'Complete ball positions size: {self.complete_ball_positions}')
    
    # Remove invalid ball positions and interpolate missing ball positions
    def clean_up_ball_positions(self):
        self.clean_ball_positions = copy.deepcopy(self.single_ball_positions)
        self.collect_stats()
        mean_velocity = np.mean(self.velocities)
        std_velocity = np.std(self.velocities)
        cutoff_velocity = mean_velocity + 2 * std_velocity
        print(f'Velocity: mean {mean_velocity}, std {std_velocity}, cutoff {cutoff_velocity}')
        while self.remove_invalid_ball_positions(cutoff_velocity):
            self.collect_stats()
        self.remove_false_hits()

    # Remove false hits - TODO
    def remove_false_hits(self):
        mean_velocity_delta = np.mean(self.velocity_deltas[2:])
        std_velocity_delta = np.std(self.velocity_deltas[2:])
        cutoff_velocity_delta = mean_velocity_delta - 3 * std_velocity_delta
        print(f'Velocity delta: mean {mean_velocity_delta}, std {std_velocity_delta}, cutoff {cutoff_velocity_delta}')

    # Collect all necessary data for further analysis
    def collect_stats(self):
        self.complete_ball_positions = self.interpolate_ball_positions(self.clean_ball_positions)
        self.complete_centre_points = self.find_centre_points(self.complete_ball_positions)
        self.velocities = self.calculate_velocities(self.complete_centre_points)
        self.velocity_vectors = self.calculate_velocity_vectors(self.complete_centre_points)
        self.velocity_deltas = self.calculate_velocity_deltas(self.velocity_vectors)

    # Interpolate missing ball positions with simple exponential deceleration
    def interpolate_ball_positions(self, ball_positions):
        bounding_boxes = [x.get(1,[]) for x in ball_positions]
        missing_ranges = self.find_missing_ranges(bounding_boxes)
        for missing_range in missing_ranges:
            self.interpolate_missing_range(missing_range, bounding_boxes)
        df = pd.DataFrame(bounding_boxes, columns=['x1','y1','x2','y2'])
        df = df.bfill()
        df = df.ffill()
        return [{1:x} for x in df.to_numpy().tolist()]

    # Find missing range between 2 known ball positions
    def find_missing_ranges(self, bounding_boxes):
        missing_ranges = []
        start_index = None
        for i, bounding_box in enumerate(bounding_boxes):
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

    # Interpolate the missing range with simple exponential deceleration
    def interpolate_missing_range(self, missing_range, bounding_boxes):
        start_bounding_box = bounding_boxes[missing_range[0] - 1]
        end_bounding_box = bounding_boxes[missing_range[1] + 1]
        start_position = get_bounding_box_center_point(start_bounding_box)
        end_position = get_bounding_box_center_point(end_bounding_box)
        steps = missing_range[1] - missing_range[0] + 3
        sequence_x = self.exponential_deceleration(start_position[0], end_position[0], steps)
        sequence_x = sequence_x[1:-1]
        sequence_y = self.exponential_deceleration(start_position[1], end_position[1], steps)
        sequence_y = sequence_y[1:-1]
        width = ((start_bounding_box[2] - start_bounding_box[0]) + (end_bounding_box[2] - end_bounding_box[0])) / 2
        height = ((start_bounding_box[3] - start_bounding_box[1]) + (end_bounding_box[3] - end_bounding_box[1])) / 2
        for i in range(len(sequence_x)):
            bounding_box = [
                sequence_x[i] - width / 2, 
                sequence_y[i] - height / 2, 
                sequence_x[i] + width / 2, 
                sequence_y[i] + height / 2]
            bounding_boxes[missing_range[0] + i] = bounding_box

    # Calculate the exponential deceleration sequence
    def exponential_deceleration(self, start, end, steps, exponent=1.2):
        if steps <= 0:
            return []
        if steps == 1:
            return [start]
        sequence = [start + (end - start) * (1 - (1 - i / (steps - 1)) ** exponent) for i in range(steps)]
        return sequence

    # Calculate velocity per frame
    def calculate_velocities(self, centre_points):
        velocities = []
        velocities.append(0)
        for i in range(1, self.frame_count):
            velocity = get_distance_between_points(centre_points[i], centre_points[i - 1])
            velocities.append(int(velocity))
        return velocities

    # Calculate delta of velocity per frame
    def calculate_velocity_deltas(self, velocities):
        velocity_deltas = []
        velocity_deltas.append(None)
        velocity_deltas.append(None)
        for i in range(2, self.frame_count):
            velocity_deltas.append(velocities[i] - velocities[i - 1])
        return velocity_deltas

    # Calculate velocity per frame with direction
    def calculate_velocity_vectors(self, centre_points):
        delta_ys = []
        delta_ys.append(None)
        for i in range(1, self.frame_count):
            delta_ys.append(centre_points[i][1] - centre_points[i - 1][1])
        velocities = []
        velocities.append(0)
        for i in range(1, self.frame_count):
            velocity = get_distance_between_points(centre_points[i], centre_points[i - 1])
            if delta_ys[i] < 0:
                velocity *= -1
            velocities.append(int(velocity))
        return velocities

    # Find centre points of ball positions
    def find_centre_points(self, ball_positions):
        bounding_boxes = [x.get(1, []) for x in ball_positions]
        centre_points = []
        for bounding_box in bounding_boxes:
            if len(bounding_box) == 0:
                centre_points.append(None)
            else:
                centre_points.append(get_bounding_box_center_point(bounding_box))
        return centre_points

    # Remove those ball positions have impossible velocity        
    def remove_invalid_ball_positions(self, cutoff_velocity):
        removed = False
        i = 1
        while i < self.frame_count:
            if self.velocities[i] < cutoff_velocity:
                i += 1
                continue
            # Found outlier
            print(f'Outlier: {self.velocities[i]}')
            if len(self.clean_ball_positions[i]) == 0:
                # Outlier caused by next detected ball position
                i += 1
                while i < self.frame_count:
                    if len(self.clean_ball_positions[i]) == 0:
                        i += 1
                        continue
                    else:
                        self.clean_ball_positions[i] = {}
                        print(f'Clear ball position: {i}')
                        break
            else:
                # Outlier is the direct result of this ball position
                self.clean_ball_positions[i] = {}
                print(f'Clear ball position: {i}')
            # Let's carry on, but jump 10 position to avoid flow on effect of the current outlier
            i += 10
            removed = True
        return removed

    def print_data_frame(self):
        df = pd.DataFrame({
            'single_centre_point': self.single_centre_points,
            'clean_centre_point': self.clean_centre_points,
            'complete_centre_point': self.complete_centre_points,
            'velocity_vector': self.velocity_vectors,
            'velocity': self.velocities,
            'velocity_delta': self.velocity_deltas,
        })
        df['frame'] = df.index
        df.to_csv('ball_trajectory_data.csv', index=False)

    def find_ball_hits(self, fps):
        df = pd.DataFrame({
            'velocity': self.velocities,
        })
        df['velocity_rolling_mean'] = df['velocity'].rolling(window = 5, min_periods = 1, center = False).mean().round().astype(int)
        peaks, _ = find_peaks(df['velocity_rolling_mean'], width=5, prominence=5)
        print(f'original peaks: {peaks}')
        troughs, _ = find_peaks(-df['velocity_rolling_mean'], prominence=5)
        print(f'original troughs: {troughs}')

        self.near_hits = self.find_near_hits()
        self.near_bounces = self.find_near_bounces(peaks, fps)
        self.far_bounces = self.find_far_bounces(troughs, fps)
        self.far_hits = self.find_far_hits()
        self.hits_and_bounces = self.sort_hits_and_bounces()
        print(f'Sorted hits and bounces: {self.hits_and_bounces}')
        self.near_hit_bounces = self.find_near_hit_bounces()
        print(f'Near hit bounces: {self.near_hit_bounces}')
        self.far_hit_bounces = self.find_far_hit_bounces()
        print(f'Far hit bounces: {self.far_hit_bounces}')

    # Find near hits by finding the highest negative velocity delta, it's a 
    # hit if at least next 10 consecutive velocity vectors are all negative
    # 37	 24	    -3.0
    # 38	-85	  -109.0
    # 39	-76	     9.0
    # 40	-59	    17.0
    # 41	-48	    11.0
    # 42	-45	     3.0
    # 43	-43	     2.0
    # 44	-43	     0.0
    # 45	-41	     2.0
    # 46	-40	     1.0
    # 47	-39	     1.0
    # 48	-36	     3.0
    # Near hit at frame 37 with lowest velocity delta at 108, and 10 
    # consecutive negative velocity vectors
    def find_near_hits(self):
        mean = np.mean(self.velocity_deltas[2:])
        std = np.std(self.velocity_deltas[2:])
        cutoff = mean - 3 * std
        print(f'Velocity delta: mean {mean}, std {std}, cutoff {cutoff}')
        candidates = []
        for i in range(2, self.frame_count):
            if self.velocity_deltas[i] < cutoff:
                candidates.append(i)
        print(f'Near hits candidates: {candidates}')
        near_hits = []
        for candidate in candidates:
            if self.all_negative_velocity(candidate, 10):
                near_hits.append(candidate)
        near_hits = [item - 1 for item in near_hits]
        print(f'Near hits: {near_hits}')
        return near_hits

    # Check consecutive negative velocity vectors
    def all_negative_velocity(self, start_index, count):
        end_index = min(start_index + count, self.frame_count)
        for i in range(start_index, end_index):
            if self.velocity_vectors[i] >= 0:
                return False
        return True

    # Find near bounces by finding the peaks of velocity rolling mean, within
    # a reasonable time before near hit, and generally towards the near player
    # after the bounce
    # 107	-26.0
    # 108	-27.0
    # 109	  3.0
    # 110	 -2.0
    # 111	 22.0
    # 112	  1.0
    # 113	  2.0
    # 114	 -5.0
    # 115	  9.0
    # 116	 -6.0
    # 117	  6.0
    # Near bounce at frame 107, with 108 has the lowest velocity delta
    def find_near_bounces(self, peaks, fps):
        print(f'frame_range: {fps}')
        candidates = []
        for peak in peaks:
            if self.peak_in_range(peak, fps):
                candidates.append(int(peak))
        print(f'Near bounce candidates: {candidates}')
        near_bounces = []
        for candidate in candidates:
            sum_velocity_delta, near_bounce = self.refine_near_bounce(candidate)
            if sum_velocity_delta > 0:
                continue
            if not near_bounce in near_bounces:
                near_bounces.append(near_bounce)
        near_bounces = [item - 1 for item in near_bounces]        
        print(f'Near bounces: {near_bounces}')
        return near_bounces

    # Check the peak is within 1 second before a near hit
    def peak_in_range(self, peak, frame_range):
        for near_hit in self.near_hits:
            if peak > near_hit - frame_range and peak < near_hit:
                return True
        if peak > self.near_hits[-1]:
            return True
        return False

    # A near bounce at lowest velocity delta, with the sum of next 10 
    # velocity deltas negative
    def refine_near_bounce(self, candidate):
        start_index = candidate
        end_index = min(start_index + 10, self.get_next_near_hit(start_index))
        min_velocity_delta_index = start_index
        sum_velocity_delta = 0
        for i in range(start_index + 1, end_index):
            if self.velocity_deltas[i] < self.velocity_deltas[min_velocity_delta_index]:
                min_velocity_delta_index = i
            sum_velocity_delta += self.velocity_deltas[i]
        return sum_velocity_delta, min_velocity_delta_index

    # Find far bounces by finding the troughs of velocity rolling mean, we also
    # assume a reasonable range the bounce could land in, and the exact bounce
    # is the frame just before the lowest velocity delta frame
    # 134	  3.0
    # 135	  1.0
    # 136	  2.0
    # 137	-10.0
    # 138	-18.0
    # 139	  6.0
    # 140	  1.0
    # 141	  1.0
    # 142	  4.0
    # 143	 -1.0
    # 144	  5.0
    # Far bounce at frame 137, with lowest velocity delta at 138
    def find_far_bounces(self, troughs, fps):
        frame_ranges = self.find_far_bounce_frame_ranges(fps)
        print(f'frame ranges: {frame_ranges}')
        candidates = []
        for trough in troughs:
            if self.trough_in_range(trough, frame_ranges):
                candidates.append(int(trough))
        print(f'Far bounce candidates: {candidates}')
        far_bounces = []
        for candidate in candidates:
            far_bounce = self.refine_far_bounce(candidate)
            if not far_bounce in far_bounces:
                far_bounces.append(far_bounce)
        far_bounces = [item - 1 for item in far_bounces]
        print(f'Far bounces: {far_bounces}')
        return far_bounces

    # Find the pair of near hit and near bounce, the far bounce is assumed to 
    # be between 1/2 second after the hit, and 1 second before the bounce
    def find_far_bounce_frame_ranges(self, fps):
        after_hit_buffer = fps / 2
        before_bounce_buffer = fps
        frame_ranges = []
        near_hits_bounces = np.concatenate((self.near_hits, self.near_bounces))
        near_hits_bounces = np.sort(near_hits_bounces)
        print(f'near hits bounces: {near_hits_bounces}')
        i = 0
        while i < len(near_hits_bounces):
            hit_index = self.find_next_near_hit_index(i, near_hits_bounces)
            if hit_index == len(near_hits_bounces):
                break
            i = hit_index + 1
            bounce_index = self.find_next_near_bounce_index(i, near_hits_bounces)
            if bounce_index == len(near_hits_bounces):
                range_start = int(near_hits_bounces[hit_index] + after_hit_buffer)
                frame_ranges.append((range_start, self.frame_count))
                break
            else:
                range_start = int(near_hits_bounces[hit_index] + after_hit_buffer)
                range_end = int(near_hits_bounces[bounce_index] - before_bounce_buffer)
                frame_ranges.append((range_start, range_end))
                i = bounce_index + 1
        return frame_ranges
    
    # From the given near_hits_bounces list, find the next index which has value in the near_hits 
    def find_next_near_hit_index(self, index, near_hits_bounces):
        for i in range(index, len(near_hits_bounces)):
            if near_hits_bounces[i] in self.near_hits:
                return i
        return len(near_hits_bounces)

    # From the given near_hits_bounces list, find the next index which has value in the near_bounces 
    def find_next_near_bounce_index(self, index, near_hits_bounces):
        for i in range(index, len(near_hits_bounces)):
            if near_hits_bounces[i] in self.near_bounces:
                return i
        return len(near_hits_bounces)
    
    # Check if the given trough is in any of the frame_ranges
    def trough_in_range(self, trough, frame_ranges):
        for frame_range in frame_ranges:
            if trough > frame_range[0] and trough < frame_range[1]:
                return True
        return False

    # Find the frame number has the lowest velocity delta value
    def refine_far_bounce(self, candidate):
        start_index = candidate
        end_index = min(start_index + 10, self.frame_count)
        min_velocity_delta_index = start_index
        for i in range(start_index + 1, end_index):
            if self.velocity_deltas[i] < self.velocity_deltas[min_velocity_delta_index]:
                min_velocity_delta_index = i
        return min_velocity_delta_index

    def find_far_hits(self):
        self.ball_shot_frame_numbers = self.find_ball_shot_frame_numbers()
        print(f'All hits: {self.ball_shot_frame_numbers}')
        candidates = []
        for hit in self.ball_shot_frame_numbers:
            if self.hit_in_near_hits_range(hit):
                continue
            candidates.append(hit)
        print(f'Far hits candidates: {candidates}')
        far_hits = []
        for candidate in candidates:
            far_hit = self.refine_far_hit(candidate)
            if far_hit is None:
                continue
            far_hits.append(far_hit)
        far_hits = [item - 1 for item in far_hits]
        print(f'Far hits: {far_hits}')
        return far_hits
    
    def refine_far_hit(self, candidate):
        start_index = candidate
        while start_index > 0 and self.velocity_vectors[start_index] >=0:
            start_index -= 1
        start_index += 1
        for i in range(start_index, start_index + 10):
            if self.velocity_vectors[i] < 0:
                return None
        return start_index

    def hit_in_near_hits_range(self, hit):
        for near_hit in self.near_hits:
            if hit > near_hit - 10 and hit < near_hit + 10:
                return True
        return False

    def find_ball_shot_frame_numbers(self):
        bounding_boxes = [x.get(1, []) for x in self.complete_ball_positions]
        df = pd.DataFrame(bounding_boxes, columns=['x1','y1','x2','y2'])
        df['mid_y'] = (df['y1'] + df['y2']) / 2
        df['mid_y_rolling_mean'] = df['mid_y'].rolling(window = 5, min_periods = 1, center = False).mean()
        df['delta_y'] = df['mid_y_rolling_mean'].diff()
        df['ball_hit'] = 0
        minimum_change_frames_for_hit = 25
        for i in range(1, len(df) - int(minimum_change_frames_for_hit * 1.2)):
            negative_change = df.at[i, 'delta_y'] > 0 and df.at[i+1, 'delta_y'] < 0
            positive_change = df.at[i, 'delta_y'] < 0 and df.at[i+1, 'delta_y'] > 0
            if negative_change or positive_change:
                change_count = 0
                # Count how many following frames have the same change
                # In this case, if 25 out of 30 frames have the same change, we consider it a hit 
                for j in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    still_negative = df.at[i, 'delta_y'] > 0 and df.at[j, 'delta_y'] < 0
                    still_positive = df.at[i, 'delta_y'] < 0 and df.at[j, 'delta_y'] > 0
                    if negative_change and still_negative:
                        change_count += 1
                    elif positive_change and still_positive:
                        change_count += 1            
                if change_count >= minimum_change_frames_for_hit:
                    df.loc[i, 'ball_hit'] = 1
        return df[df['ball_hit'] == 1].index.tolist()



    def get_next_near_hit(self, index):
        for near_hit in self.near_hits:
            if index < near_hit:
                return near_hit
        return self.frame_count - 1

    def find_near_hit_bounces(self):
        near_hit_bounces = []
        for i in range(2, len(self.hits_and_bounces) - 1, 4):
            hit = self.hits_and_bounces[i]
            bounce = self.hits_and_bounces[i + 1]
            if hit is not None and bounce is not None:
                near_hit_bounces.append((hit, bounce))
        return near_hit_bounces
    
    def find_far_hit_bounces(self):
        far_hit_bounces = []
        for i in range(0, len(self.hits_and_bounces) - 1, 4):
            hit = self.hits_and_bounces[i]
            bounce = self.hits_and_bounces[i + 1]
            if hit is not None and bounce is not None:
                far_hit_bounces.append((hit, bounce))
        return far_hit_bounces
    
    def sort_hits_and_bounces(self):
        candidates = np.concatenate((self.near_hits, self.far_hits, self.near_bounces, self.far_bounces))
        candidates = np.sort(candidates)
        print(f'Sort hits and bounces: {candidates}')
        hits_and_bounces = []
        far_hit = True
        near_bounce = False
        near_hit = False
        far_bounce = False
        i = 0
        while i < len(candidates):
            hit_or_bounce = candidates[i]
            if far_hit:
                if hit_or_bounce in self.far_hits:
                    hits_and_bounces.append(int(hit_or_bounce))
                    i += 1
                else:
                    hits_and_bounces.append(None)
                far_hit = False
                near_bounce = True
            elif near_bounce:
                if hit_or_bounce in self.near_bounces:
                    hits_and_bounces.append(int(hit_or_bounce))
                    i += 1
                else:
                    hits_and_bounces.append(None)
                near_bounce = False
                near_hit = True
            elif near_hit:
                if hit_or_bounce in self.near_hits:
                    hits_and_bounces.append(int(hit_or_bounce))
                    i += 1
                else:
                    hits_and_bounces.append(None)
                near_hit = False
                far_bounce = True
            elif far_bounce:
                if hit_or_bounce in self.far_bounces:
                    hits_and_bounces.append(int(hit_or_bounce))
                    i += 1
                else:
                    hits_and_bounces.append(None)
                far_bounce = False
                far_hit = True
        return hits_and_bounces    

    # Draw bounding boxes on the frames
    def draw(self, input_frames):
        output_frames = []
        for frame, ball_position in zip(input_frames, self.complete_ball_positions):
            for track_id, bounding_box in ball_position.items():
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
                cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            output_frames.append(frame)
        return output_frames
