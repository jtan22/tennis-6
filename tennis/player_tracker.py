from ultralytics import YOLO
import cv2
import pickle
import torch
from .utils import get_bottom_line_center_point

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available else 'cpu')
        self.model.to(device)

    # Detect players in a list of frames
    def dectect_person_positions_all_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None:
            # Read frames from stub
            with open(stub_path, 'rb') as f:
                self.person_positions = pickle.load(f)
            return
            
        self.person_positions = []
        for frame in frames:
            self.person_positions.append(self.detect_person_positions_per_frame(frame))

        if stub_path is not None:
            # Save frames to stub
            with open(stub_path, 'wb') as f:
                pickle.dump(self.person_positions, f)
            return

    # Detect players in a single frame
    # Returns a dictionary with track IDs as keys and bounding boxes as values
    # Bounding box format: [x1, y1, x2, y2]
    # x1, y1 are the top-left coordinates and x2, y2 are the bottom-right coordinates
    def detect_person_positions_per_frame(self, frame):
        # Perform tracking on the frame
        # The model will return a list of results, we take the first one
        results = self.model.track(frame, persist=True)[0]
        # The 'names' attribute contains a dictionary mapping class IDs to class names
        class_id_name_dict = results.names
        person_positions_per_frame = {}
        # The results will contain the bounding boxes, track IDs, and class IDs
        for box in results.boxes:
            # The track IDs are stored in the 'id' attribute of the boxes
            track_id = int(box.id.tolist()[0])
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
    
    def filter_out_non_players(self, court_keypoints):
        person_positions_first_frame = self.person_positions[0]
        chosen_players = self.choose_players(court_keypoints, person_positions_first_frame)
        self.player_positions = []
        for person_positions_per_frame in self.person_positions:
            player_positions_per_frame = {track_id: bounding_box 
                    for track_id, bounding_box in person_positions_per_frame.items() 
                    if track_id in chosen_players}
            self.player_positions.append(player_positions_per_frame)

    # Choose 2 persons have the shortest distance to any of the baselines
    def choose_players(self, court_keypoints, person_positions_per_frame):
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
        return first_2_track_ids

    # Draw bounding boxes on the frames
    def draw(self, input_frames):
        output_frames = []
        for frame, player_positions_per_frame in zip(input_frames, self.player_positions):
            for track_id, bounding_box in player_positions_per_frame.items():
                x1, y1, x2, y2 = bounding_box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Draw the bounding box on the frame
                # The bounding box is drawn in green color with a thickness of 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw the track ID above the bounding box
                # The track ID is drawn in green color with a font scale of 1
                # and a thickness of 2
                # The text is drawn at the top-left corner of the bounding box
                # The text is drawn slightly above the bounding box
                cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            output_frames.append(frame)
        return output_frames
