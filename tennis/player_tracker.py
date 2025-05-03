from typing import Dict, List, Tuple
from ultralytics import YOLO
import cv2
import pickle
import torch
import pandas as pd
from .utils import get_bottom_line_center_point

class PlayerTracker:

    def __init__(self):
        self.stub_path = 'tracker_stubs/player_detections.pkl'
        self.player_positions_path = 'analysis/player_positions.csv'

    # Detect persons in a list of frames
    def dectect_person_positions(self, frames, model_path) -> None:
        model = YOLO(model_path)
        model.to(torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available else 'cpu'))

        person_positions = []
        for frame in frames:
            person_positions.append(self._detect_person_positions_per_frame(frame, model))

        # Save frames to stub
        with open(self.stub_path, 'wb') as f:
            pickle.dump(person_positions, f)
    
    # Detect players in a single frame
    # Returns a dictionary with track IDs as keys and bounding boxes as values
    # Bounding box format: [x1, y1, x2, y2]
    # x1, y1 are the top-left coordinates and x2, y2 are the bottom-right coordinates
    def _detect_person_positions_per_frame(self, frame, model):
        # Perform tracking on the frame
        # The model will return a list of results, we take the first one
        results = model.track(frame, persist=True)[0]
        # The 'names' attribute contains a dictionary mapping class IDs to class names
        class_id_name_dict = results.names
        person_positions_per_frame = {}
        # The results will contain the bounding boxes, track IDs, and class IDs
        for box in results.boxes: # type: ignore
            # The track IDs are stored in the 'id' attribute of the boxes
            track_id = int(box.id.tolist()[0]) # type: ignore
            # The bounding boxes are stored in the 'xyxy' attribute of the boxes
            # The 'xyxy' attribute contains a list of lists, where each inner list
            # contains the coordinates of the bounding box in the format [x1, y1, x2, y2]
            bounding_box = box.xyxy.tolist()[0]
            # The class IDs are stored in the 'cls' attribute of the boxes
            class_id = box.cls.tolist()[0]
            class_name = class_id_name_dict[class_id]
            if class_name == 'person':
                person_positions_per_frame[track_id] = bounding_box
        return person_positions_per_frame
    
    def _load_person_positions_from_stub(self):
        # Read frames from stub
        with open(self.stub_path, 'rb') as f:
            return pickle.load(f)

    def find_player_positions(self, court_keypoints):
        person_positions = self._load_person_positions_from_stub()
        near_player_id, far_player_id = self._choose_players(court_keypoints, person_positions[0])
        near_player_positions = []
        far_player_positions = []
        for person_positions_per_frame in person_positions:
            near_player_positions_per_frame = {track_id: [int(x) for x in bounding_box]
                    for track_id, bounding_box in person_positions_per_frame.items() 
                    if track_id == near_player_id}
            near_player_positions.append(near_player_positions_per_frame)
            far_player_positions_per_frame = {track_id: [int(x) for x in bounding_box]
                    for track_id, bounding_box in person_positions_per_frame.items() 
                    if track_id == far_player_id}
            far_player_positions.append(far_player_positions_per_frame)

        df = pd.DataFrame({
            'frame': range(len(near_player_positions)),
            'near_player_position': near_player_positions,
            'far_player_position': far_player_positions,
        })
        df.to_csv(self.player_positions_path, index=False)

    def load_player_positions(self) -> Tuple[List[Dict[int, Tuple[int, int, int, int]]], List[Dict[int, Tuple[int, int, int, int]]]]:
        # Load player positions from CSV file
        df = pd.read_csv(self.player_positions_path)
        near_player_positions = df['near_player_position'].tolist()
        far_player_positions = df['far_player_position'].tolist()
        near_player_positions = [eval(x) for x in near_player_positions]
        far_player_positions = [eval(x) for x in far_player_positions]
        return near_player_positions, far_player_positions

    # Choose 2 persons have the shortest distance to any of the baselines
    def _choose_players(self, court_keypoints, person_positions_per_frame):
        distances = []
        for track_id, bounding_box in person_positions_per_frame.items():
            person_bottom_center = get_bottom_line_center_point(bounding_box)
            if person_bottom_center[0] < court_keypoints[2][0] or person_bottom_center[0] > court_keypoints[3][0]:
                continue
            min_distance = abs(person_bottom_center[1] - court_keypoints[0][1])
            distance = abs(person_bottom_center[1] - court_keypoints[2][1])
            if distance < min_distance:
                min_distance = distance
            distances.append((track_id, min_distance))
        
        # Sort the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 track_ids
        first_2_track_ids = [distances[0][0], distances[1][0]]
        bounding_box1 = person_positions_per_frame[first_2_track_ids[0]]
        bounding_box2 = person_positions_per_frame[first_2_track_ids[1]]
        if bounding_box1[1] > bounding_box2[1]:
            far_player_id = first_2_track_ids[1]
            near_player_id = first_2_track_ids[0]
        else:
            far_player_id = first_2_track_ids[0]
            near_player_id = first_2_track_ids[1]
        print(f'Near player: {near_player_id}, far player: {far_player_id}')
        return near_player_id, far_player_id

    # Draw bounding boxes on the frames
    def draw(self, input_frames):
        near_player_positions, far_player_positions = self.load_player_positions()
        output_frames = []
        color_red = (0, 0, 255)
        color_blue = (255, 0, 0)
        for i, frame in enumerate(input_frames):
            for track_id, bounding_box in near_player_positions[i].items():
                x1, y1, x2, y2 = bounding_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_red, 2)
            for track_id, bounding_box in far_player_positions[i].items():
                x1, y1, x2, y2 = bounding_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_blue, 2)
            output_frames.append(frame)
        return output_frames
