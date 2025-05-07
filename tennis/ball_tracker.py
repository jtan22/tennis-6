from csv import Error
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pickle
import torch
from ultralytics import YOLO
import copy

from .utils import get_distance_between_point_and_line
from .bounding_box import BoundingBox

class BallTracker:
    
    def __init__(self, model_path: str):
        self.stub_path = 'tracker_stubs/ball_detections.pkl'
        self.ball_positions_path = 'analysis/ball_positions.csv'
        self.model = YOLO(model_path)
        self.model.to('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        
    def detect_ball_positions(self, frames: List[np.ndarray]) -> None:
        ball_positions = []
        for i, frame in enumerate(frames):
            positions = self.detect_ball_positions_per_frame(frame)
            ball_positions.append(positions)

        with open(self.stub_path, 'wb') as f:
            pickle.dump(ball_positions, f)

    def _load_ball_positions_from_stub(self) -> List[Dict[int, Tuple[int, int, int, int]]]:
        try:
            with open(self.stub_path, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.PickleError, EOFError) as e:
            raise Error(f"Error loading stub file: {e}")

    def detect_ball_positions_per_frame(self, frame: np.ndarray) -> Dict[int, Tuple[int, int, int, int]]:
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty or invalid")

        try:
            results = self.model.predict(frame, conf=0.15, verbose=False)[0]
            ball_positions = {}            
            for i, box in enumerate(results.boxes, start=1): # type: ignore
                # Convert to list and round to integer coordinates
                bounding_box = [int(coord) for coord in box.xyxy.tolist()[0]]
                ball_positions[i] = bounding_box
            return ball_positions
        except Exception as e:
            raise Error(f"Error detecting ball in frame: {e}")

    def synthesize_ball_positions(self, max_iterations: int = 5) -> None:
        multiple_ball_positions = self._load_ball_positions_from_stub()
            
        # First, reduce to one ball per frame
        single_ball_positions = self._remove_extra_balls_detected(multiple_ball_positions)
        # All outliers removed from single_ball_positions
        clean_ball_positions = copy.deepcopy(single_ball_positions)
        
        # Prepare initial data
        df = self._build_complete_ball_positions(clean_ball_positions)
        if df.empty:
            raise Error("Warning: No valid ball positions detected")
            
        # Calculate outlier threshold based on velocity
        velocity_stats = df['velocity'].agg(['mean', 'std']).to_dict()
        cutoff_velocity = velocity_stats['mean'] + 2 * velocity_stats['std']
        print(f'Velocity: mean {velocity_stats['mean']:.2f}, std {velocity_stats['std']:.2f}, cutoff {cutoff_velocity:.2f}')
        
        # Iteratively remove outliers
        iteration_count = 0
        while self._remove_invalid_ball_positions(cutoff_velocity, df, clean_ball_positions) and iteration_count < max_iterations:
            df = self._build_complete_ball_positions(clean_ball_positions)
            iteration_count += 1
            
        print(f"Completed ball position synthesis after {iteration_count} iterations")
        df.insert(0, 'frame', range(len(clean_ball_positions)))
        df.insert(1, 'multi_ball_position', multiple_ball_positions)
        df.insert(2, 'single_ball_position', single_ball_positions)
        df.insert(3, 'clean_ball_position', clean_ball_positions)
        df.to_csv(self.ball_positions_path, index=False)

    def _remove_extra_balls_detected(self, 
            multiple_ball_positions: List[Dict[int, Tuple[int, int, int, int]]]) -> List[Tuple[int, int, int, int] | None]:
        single_ball_positions = []
        
        for i, ball_positions in enumerate(multiple_ball_positions):
            # If there's only one or no ball detected, keep as is
            if len(ball_positions) == 0:
                single_ball_positions.append(None)
                continue
            if len(ball_positions) == 1:
                single_ball_positions.append(ball_positions[1])
                continue
                
            # Get context from adjacent frames
            last_ball_position = self._find_adjacent_ball_position(multiple_ball_positions, i, look_backward=True)
            next_ball_position = self._find_adjacent_ball_position(multiple_ball_positions, i, look_backward=False)
            
            # If we don't have enough context, keep the first ball
            if last_ball_position is None or next_ball_position is None:
                single_ball_positions.append(list(ball_positions.values())[0])
                continue
                
            # Find the most likely ball based on trajectory
            best_ball = self._find_best_ball_position(last_ball_position, ball_positions, next_ball_position)
            single_ball_positions.append(best_ball)
            
        return single_ball_positions

    def _find_adjacent_ball_position(self, 
                                     multiple_ball_positions: List[Dict[int, Tuple[int, int, int, int]]], 
                                     index: int, 
                                     look_backward: bool = True) -> Optional[Tuple[int, int, int, int]]:
        if look_backward:
            if index == 0:
                return None
                
            search_range = range(index - 1, -1, -1)
        else:
            if index >= len(multiple_ball_positions) - 1:
                return None
                
            search_range = range(index + 1, len(multiple_ball_positions))
            
        for i in search_range:
            positions = multiple_ball_positions[i]
            if len(positions) == 1 and 1 in positions:
                return positions[1]
                
        return None

    def _find_best_ball_position(self, 
                              last_ball_position: Tuple[int, int, int, int],
                              ball_positions: Dict[int, Tuple[int, int, int, int]],
                              next_ball_position: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        line_start = BoundingBox.from_list(last_ball_position).center
        line_end = BoundingBox.from_list(next_ball_position).center
        
        # Calculate trajectory deviation for each candidate
        best_ball_id = None
        min_distance = float('inf')
        
        for ball_id, ball_position in ball_positions.items():
            point = BoundingBox.from_list(ball_position).center
            distance = get_distance_between_point_and_line(point, line_start, line_end)
            
            if distance < min_distance:
                min_distance = distance
                best_ball_id = ball_id
                
        # Fallback to first ball if something went wrong
        if best_ball_id is None:
            best_ball_id = next(iter(ball_positions))
            
        return ball_positions[best_ball_id]

    def _build_complete_ball_positions(self, clean_ball_positions: List[Tuple[int, int, int, int] | None]) -> pd.DataFrame:
        # Fill in missing positions
        complete_ball_positions = self._interpolate_ball_positions(copy.deepcopy(clean_ball_positions))
        
        # Create DataFrame with all position data
        df = pd.DataFrame({
            'complete_ball_position': complete_ball_positions,
        })
        
        # Extract center points
        df['centre_point'] = df['complete_ball_position'].apply(lambda x: BoundingBox.from_list(x).center)
        df[['x', 'y']] = pd.DataFrame(df['centre_point'].tolist(), index=df.index)
        
        # Calculate motion metrics
        df['x_diff'] = df['x'].diff().fillna(0).astype(int)
        df['y_diff'] = df['y'].diff().fillna(0).astype(int)
        
        # Velocity calculations
        df['velocity'] = np.sqrt(df['x_diff']**2 + df['y_diff']**2).round(2)
        
        return df

    def _interpolate_ball_positions(self, ball_positions: List[Tuple[int, int, int, int] | None]) -> List[Tuple[int, int, int, int]]:
        # Find and interpolate missing ranges
        missing_ranges = self._find_missing_ranges(ball_positions)
        
        for start_idx, end_idx in missing_ranges:
            self._interpolate_missing_range((start_idx, end_idx), ball_positions)
        
        # Create DataFrame for easier handling of remaining missing values
        df = pd.DataFrame({'ball_position': ball_positions})
        df['ball_position'] = df['ball_position'].ffill().bfill()
        
        # Convert back to the original format
        return df['ball_position'].tolist()

    def _find_missing_ranges(self, bounding_boxes: List[Tuple[int, int, int, int] | None]) -> List[Tuple[int, int]]:
        missing_ranges = []
        start_index = None
        
        for i, box in enumerate(bounding_boxes):
            if not box:  # Empty box
                if start_index is None:
                    start_index = i
            else:
                if start_index is not None:
                    missing_ranges.append((start_index, i - 1))
                    start_index = None
                    
        # Handle case where the last positions are missing
        if start_index is not None:
            missing_ranges.append((start_index, len(bounding_boxes) - 1))
            
        # Filter out ranges that can't be interpolated
        valid_ranges = []
        for start, end in missing_ranges:
            if start > 0 and end < len(bounding_boxes) - 1 and bounding_boxes[start - 1] and bounding_boxes[end + 1]:
                valid_ranges.append((start, end))
                
        return valid_ranges

    def _interpolate_missing_range(self, missing_range: Tuple[int, int], bounding_boxes: List[Tuple[int, int, int, int] | None]) -> None:
        start_idx, end_idx = missing_range
        
        start_bounding_box = bounding_boxes[start_idx - 1]
        end_bounding_box = bounding_boxes[end_idx + 1]
        if start_bounding_box is None or end_bounding_box is None:
            raise Error(f'None bounding box: {start_bounding_box}, {end_bounding_box}')
        
        # Get bounding boxes before and after the missing range
        start_box = BoundingBox.from_list(start_bounding_box)
        end_box = BoundingBox.from_list(end_bounding_box)
            
        # Calculate total steps including start and end points
        steps = end_idx - start_idx + 3
        
        # Generate interpolated sequences for x and y coordinates
        sequence_x = self._generate_interpolation_sequence(start_box.center[0], end_box.center[0], steps)
        sequence_y = self._generate_interpolation_sequence(start_box.center[1], end_box.center[1], steps)
        
        # Remove first and last points (already known)
        sequence_x = sequence_x[1:-1]
        sequence_y = sequence_y[1:-1]
        
        # Calculate average box dimensions
        avg_width = (start_box.width + end_box.width) / 2
        avg_height = (start_box.height + end_box.height) / 2
        
        # Create and insert interpolated boxes
        for i, (x, y) in enumerate(zip(sequence_x, sequence_y)):
            new_box = BoundingBox.from_center_and_size(x, y, avg_width, avg_height)
            bounding_boxes[start_idx + i] = new_box.to_list()

    def _generate_interpolation_sequence(self, start: float, end: float, steps: int, 
                                        exponent: float = 1.2) -> List[float]:
        """
        Generate a sequence with exponential deceleration between two points.
        
        Args:
            start: Starting value
            end: Ending value
            steps: Number of points in sequence
            exponent: Controls the shape of the curve (>1 for deceleration)
            
        Returns:
            List of interpolated values
        """
        if steps <= 1:
            return [start]
            
        # Generate normalized sequence with exponential weighting
        return [
            start + (end - start) * (1 - (1 - i / (steps - 1)) ** exponent)
            for i in range(steps)
        ]

    def _remove_invalid_ball_positions(self, 
            cutoff_velocity: float, 
            df: pd.DataFrame, 
            clean_ball_positions: List[Tuple[int, int, int, int] | None]) -> bool:
        removed = False
        i = 1

        frame_count = len(df)
        while i < frame_count:
            # Skip if velocity is within acceptable range
            if i >= len(df) or df.loc[i, 'velocity'] < cutoff_velocity: # type: ignore
                i += 1
                continue
                
            # Found outlier
            print(f'Outlier at frame {i}: velocity = {df.loc[i, "velocity"]:.2f}')
            
            # Determine action based on whether this is an interpolated position
            if not clean_ball_positions[i]:
                # Find next non-interpolated position
                next_i = i + 1
                while next_i < frame_count:
                    if not clean_ball_positions[next_i]:
                        next_i += 1
                        continue
                    else:
                        # Clear the next position as it's causing the issue
                        clean_ball_positions[next_i] = None
                        print(f'Cleared ball position at frame: {next_i}')
                        break
                        
                # Skip ahead to avoid removing multiple related positions
                i = next_i + 1
            else:
                # Clear current position
                clean_ball_positions[i] = None
                print(f'Cleared ball position at frame: {i}')
                # Skip ahead
                i += 10
                
            removed = True
            
        return removed

    def load_ball_positions_df(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.ball_positions_path)
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            raise Error(f"Error loading ball positions: {e}")

    def load_complete_ball_positions(self) -> List[Tuple[int, int, int, int]]:
        return [eval(x) for x in self.load_ball_positions_df()['complete_ball_position'].tolist()]

    def draw(self, input_frames: List[np.ndarray]) -> List[np.ndarray]:
        df = self.load_ball_positions_df()
        complete_ball_positions = [eval(x) for x in df['complete_ball_position'].tolist()]
        output_frames = []
        
        for i in range(len(input_frames)):
            # Create a copy to avoid modifying original frames
            frame_copy = input_frames[i].copy()
            ball_position = complete_ball_positions[i]
            
            # Create bounding box for easier handling
            box = BoundingBox.from_list(ball_position)
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
            
            # Add velocity information
            velocity = df.loc[i, 'velocity']
            # Add position information
            x, y = df.loc[i, 'x'], df.loc[i, 'y']
            cv2.putText(
                frame_copy, f"({x}, {y})", (box.x2 + 10, box.y1 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
            cv2.putText(
                frame_copy, f"v: {velocity:.1f}", (box.x2 + 10, box.y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
                        
            output_frames.append(frame_copy)
            
        return output_frames

    def draw_positions(self, frame: np.ndarray, positions: Dict[int, Tuple[int, int, int, int]]) -> None:
        color_green = (0, 255, 0)
        for id, bounding_box in positions.items():
            x1, y1, x2, y2 = bounding_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_green, 2)
            cv2.putText(frame, f"{id}", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_green, 2, cv2.LINE_AA)
    