from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pickle
import torch
from ultralytics import YOLO

from .utils import get_distance_between_point_and_line
from .bounding_box import BoundingBox

class BallTracker:
    """
    Class to track a ball in video frames using YOLOv8 model.
    
    This class detects and tracks balls across video frames, handling multiple detections,
    interpolating missing positions, and filtering outliers.
    """
    
    def __init__(self, model_path: Union[str, Path], confidence_threshold: float = 0.15):
        """
        Initialize the BallTracker with the specified model.
        
        Args:
            model_path: Path to the YOLOv8 model
            confidence_threshold: Detection confidence threshold (0-1)
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Select appropriate device
        self.device = (
            'mps' if torch.backends.mps.is_available() else
            'cuda' if torch.cuda.is_available() else
            'cpu'
        )
        self.model.to(self.device)
        
        # Initialize tracking data structures
        self.multiple_ball_positions: List[Dict[int, List[int]]] = []
        self.single_ball_positions: List[Dict[int, List[int]]] = []
        self.clean_ball_positions: List[Dict[int, List[int]]] = []
        self.complete_ball_positions: List[Dict[int, List[int]]] = []
        self.frame_count: int = 0
        self.df: Optional[pd.DataFrame] = None
        
        print(f"BallTracker initialized using {self.device} device")

    def detect_ball_positions_all_frames(self, 
                                        frames: List[np.ndarray],
                                        read_from_stub: bool = False,
                                        stub_path: Optional[Union[str, Path]] = None) -> None:
        """
        Detect ball positions in all frames or load from a saved stub file.
        
        Args:
            frames: List of video frames as numpy arrays
            read_from_stub: Whether to read from a saved stub file
            stub_path: Path to the stub file
        """
        if read_from_stub and stub_path is not None:
            if self._load_from_stub(stub_path):
                return
                
        self._detect_frames(frames)
        
        if stub_path is not None:
            self._save_to_stub(stub_path)

    def _load_from_stub(self, stub_path: Union[str, Path]) -> bool:
        """
        Load ball positions from a stub file.
        
        Args:
            stub_path: Path to the stub file
            
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            with open(stub_path, 'rb') as f:
                self.multiple_ball_positions = pickle.load(f)
            self.frame_count = len(self.multiple_ball_positions)
            print(f'Loaded {self.frame_count} frames of ball positions from stub')
            return True
        except (FileNotFoundError, pickle.PickleError, EOFError) as e:
            print(f"Error loading stub file: {e}")
            return False

    def _save_to_stub(self, stub_path: Union[str, Path]) -> bool:
        """
        Save ball positions to a stub file.
        
        Args:
            stub_path: Path to save the stub file
            
        Returns:
            True if successfully saved, False otherwise
        """
        try:
            # Ensure directory exists
            Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(stub_path, 'wb') as f:
                pickle.dump(self.multiple_ball_positions, f)
            print(f'Saved {self.frame_count} frames of ball positions to stub')
            return True
        except IOError as e:
            print(f"Error saving stub file: {e}")
            return False

    def _detect_frames(self, frames: List[np.ndarray]) -> None:
        """
        Process all frames to detect ball positions.
        
        Args:
            frames: List of video frames as numpy arrays
        """
        self.multiple_ball_positions = []
        
        for i, frame in enumerate(frames):
            positions = self._detect_ball_positions_per_frame(frame)
            self.multiple_ball_positions.append(positions)
            
            # Log progress for large videos
            if i > 0 and i % 100 == 0:
                print(f'Processed {i}/{len(frames)} frames')
                
        self.frame_count = len(self.multiple_ball_positions)
        print(f'Detected balls in {self.frame_count} frames')

    def _detect_ball_positions_per_frame(self, frame: np.ndarray) -> Dict[int, List[int]]:
        """
        Detect ball positions in a single frame.
        
        Args:
            frame: A video frame as numpy array
            
        Returns:
            Dictionary with track IDs as keys and bounding boxes [x1, y1, x2, y2] as values
        """
        if frame is None or frame.size == 0:
            return {}
            
        try:
            results = self.model.predict(frame, conf=self.confidence_threshold)[0]
            ball_positions = {}
            
            for i, box in enumerate(results.boxes, start=1):
                # Convert to list and round to integer coordinates
                bounding_box = [int(coord) for coord in box.xyxy.tolist()[0]]
                ball_positions[i] = bounding_box
                
            return ball_positions
        except Exception as e:
            print(f"Error detecting ball in frame: {e}")
            return {}

    def synthesize_ball_positions(self, max_iterations: int = 5) -> None:
        """
        Process detected ball positions to create a clean tracking sequence.
        
        Args:
            max_iterations: Maximum number of outlier removal iterations
        """
        if not self.multiple_ball_positions:
            print("No ball positions to synthesize")
            return
            
        # First, reduce to one ball per frame
        self.single_ball_positions = self._remove_extra_balls_detected()
        # All outliers removed from single_ball_positions
        self.clean_ball_positions = self.single_ball_positions.copy()
        
        # Prepare initial data
        self.df = self._prepare_data()
        if self.df.empty:
            print("Warning: No valid ball positions detected")
            return
            
        # Calculate outlier threshold based on velocity
        velocity_stats = self.df['velocity'].agg(['mean', 'std']).to_dict()
        cutoff_velocity = velocity_stats['mean'] + 2 * velocity_stats['std']
        print(f'Velocity: mean {velocity_stats['mean']:.2f}, std {velocity_stats['std']:.2f}, cutoff {cutoff_velocity:.2f}')
        
        # Iteratively remove outliers
        iteration_count = 0
        while self._remove_invalid_ball_positions(cutoff_velocity) and iteration_count < max_iterations:
            self.df = self._prepare_data()
            iteration_count += 1
            
        print(f"Completed ball position synthesis after {iteration_count} iterations")

    def _remove_extra_balls_detected(self) -> List[Dict[int, List[int]]]:
        """
        Select the best ball from multiple detections in each frame.
        
        Returns:
            List of dictionaries with one ball per frame
        """
        single_ball_positions = []
        
        for i, ball_positions in enumerate(self.multiple_ball_positions):
            # If there's only one or no ball detected, keep as is
            if len(ball_positions) <= 1:
                single_ball_positions.append(ball_positions.copy())
                continue
                
            # Get context from adjacent frames
            last_ball_position = self._find_adjacent_ball_position(i, look_backward=True)
            next_ball_position = self._find_adjacent_ball_position(i, look_backward=False)
            
            # If we don't have enough context, keep the first ball
            if last_ball_position is None or next_ball_position is None:
                single_ball_positions.append({1: list(ball_positions.values())[0]})
                continue
                
            # Find the most likely ball based on trajectory
            best_ball = self._find_best_ball_position(last_ball_position, ball_positions, next_ball_position)
            single_ball_positions.append({1: best_ball})
            
        return single_ball_positions

    def _find_adjacent_ball_position(self, index: int, look_backward: bool = True) -> Optional[List[int]]:
        """
        Find the nearest ball position in adjacent frames.
        
        Args:
            index: Current frame index
            look_backward: Whether to look for previous (True) or next (False) frames
            
        Returns:
            Bounding box of the ball if found, None otherwise
        """
        if look_backward:
            if index == 0:
                return None
                
            search_range = range(index - 1, -1, -1)
        else:
            if index >= len(self.multiple_ball_positions) - 1:
                return None
                
            search_range = range(index + 1, len(self.multiple_ball_positions))
            
        for i in search_range:
            positions = self.multiple_ball_positions[i]
            if len(positions) == 1 and 1 in positions:
                return positions[1]
                
        return None

    def _find_best_ball_position(self, 
                              last_ball_position: List[int],
                              ball_positions: Dict[int, List[int]],
                              next_ball_position: List[int]) -> List[int]:
        """
        Find the most likely ball position based on trajectory.
        
        Args:
            last_ball_position: Previous ball position
            ball_positions: Current frame's ball positions
            next_ball_position: Next frame's ball position
            
        Returns:
            Most likely bounding box for the ball
        """
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

    def _prepare_data(self) -> pd.DataFrame:
        """
        Process ball position data and calculate derived metrics.
        
        Args:
            clean_ball_positions: List of cleaned ball positions
            
        Returns:
            DataFrame with processed ball data
        """
        if not self.clean_ball_positions:
            return pd.DataFrame()
            
        # Fill in missing positions
        self.complete_ball_positions = self._interpolate_ball_positions(self.clean_ball_positions)
        
        # Helper function to safely extract center point
        def safe_center_point(pos_dict):
            if not pos_dict or 1 not in pos_dict:
                return (0, 0)
            return BoundingBox.from_list(pos_dict[1]).center
        
        # Create DataFrame with all position data
        data = {
            'multi_ball_position': self.multiple_ball_positions,
            'single_ball_position': self.single_ball_positions,
            'clean_ball_position': self.clean_ball_positions,
            'complete_ball_position': self.complete_ball_positions,
            'frame': range(len(self.clean_ball_positions)),
        }
        
        df = pd.DataFrame(data)
        
        # Extract center points
        df['centre_point'] = df['complete_ball_position'].apply(safe_center_point)
        df[['x', 'y']] = pd.DataFrame(df['centre_point'].tolist(), index=df.index)
        
        # Calculate motion metrics
        self._calculate_motion_metrics(df)
        
        # Save data for analysis (optional)
        try:
            df.to_csv('ball_data_frame.csv', index=False)
        except IOError as e:
            print(f"Warning: Could not save DataFrame to CSV: {e}")
            
        return df

    def _calculate_motion_metrics(self, df: pd.DataFrame) -> None:
        """
        Calculate motion metrics from position data.
        
        Args:
            df: DataFrame with position data to modify in-place
        """
        # Basic differentials
        df['x_diff'] = df['x'].diff().fillna(0).astype(int)
        df['y_diff'] = df['y'].diff().fillna(0).astype(int)
        
        # Velocity calculations
        df['velocity'] = np.sqrt(df['x_diff']**2 + df['y_diff']**2).round(2)
        
        # Signed velocity (positive when moving down, negative when moving up)
        df['velocity_vector'] = np.where(
            df['y_diff'] >= 0,
            df['velocity'],
            -df['velocity']
        )
        
        # Calculate smoothed metrics
        window_size = min(5, len(df))
        if window_size > 0:
            rolling_args = {'window': window_size, 'min_periods': 1, 'center': False}
            
            df['y_rolling_mean'] = df['y'].rolling(**rolling_args).mean().round(1)
            df['y_rolling_mean_delta'] = df['y_rolling_mean'].diff().round(1)
            df['velocity_vector_delta'] = df['velocity_vector'].diff().round(2)
            df['velocity_rolling_mean'] = df['velocity'].rolling(**rolling_args).mean().round(1)
            
            # Acceleration (second derivative)
            df['acceleration'] = df['velocity_vector'].diff().rolling(**rolling_args).mean().round(2)

    def _interpolate_ball_positions(self, ball_positions: List[Dict[int, List[int]]]) -> List[Dict[int, List[int]]]:
        """
        Fill in missing ball positions using interpolation.
        
        Args:
            ball_positions: List of dictionaries with ball positions
            
        Returns:
            Complete list with interpolated values for missing positions
        """
        # Extract bounding boxes as lists for easier manipulation
        bounding_boxes = []
        for pos in ball_positions:
            if pos and 1 in pos:
                bounding_boxes.append(pos[1])
            else:
                bounding_boxes.append([])
                
        # Find and interpolate missing ranges
        missing_ranges = self._find_missing_ranges(bounding_boxes)
        
        for start_idx, end_idx in missing_ranges:
            self._interpolate_missing_range((start_idx, end_idx), bounding_boxes)
            
        # Create DataFrame for easier handling of remaining missing values
        df = pd.DataFrame(bounding_boxes, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Fill any remaining gaps with forward/backward fill
        filled_df = df.bfill().ffill()
        
        # Convert back to the original format
        return [{1: box} if not pd.isna(box[0]) else {} for box in filled_df.values.tolist()]

    def _find_missing_ranges(self, bounding_boxes: List[List[int]]) -> List[Tuple[int, int]]:
        """
        Find contiguous ranges of missing ball positions.
        
        Args:
            bounding_boxes: List of bounding boxes where empty lists indicate missing positions
            
        Returns:
            List of (start_index, end_index) tuples for missing ranges
        """
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

    def _interpolate_missing_range(self, missing_range: Tuple[int, int], bounding_boxes: List[List[int]]) -> None:
        """
        Interpolate a range of missing ball positions.
        
        Args:
            missing_range: Tuple of (start_index, end_index)
            bounding_boxes: List of bounding boxes to modify in-place
        """
        start_idx, end_idx = missing_range
        
        # Get bounding boxes before and after the missing range
        try:
            start_box = BoundingBox.from_list(bounding_boxes[start_idx - 1])
            end_box = BoundingBox.from_list(bounding_boxes[end_idx + 1])
        except (IndexError, ValueError):
            return
            
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

    def _remove_invalid_ball_positions(self, cutoff_velocity: float) -> bool:
        """
        Remove ball positions with implausible velocities.
        
        Args:
            cutoff_velocity: Maximum allowed velocity
            
        Returns:
            Boolean indicating whether any positions were removed
        """
        if self.df is None or self.df.empty:
            return False
            
        removed = False
        i = 1
        
        while i < self.frame_count:
            # Skip if velocity is within acceptable range
            if i >= len(self.df) or self.df.loc[i, 'velocity'] < cutoff_velocity:
                i += 1
                continue
                
            # Found outlier
            print(f'Outlier at frame {i}: velocity = {self.df.loc[i, "velocity"]:.2f}')
            
            # Determine action based on whether this is an interpolated position
            if not self.df.loc[i, 'clean_ball_position']:
                # Find next non-interpolated position
                next_i = i + 1
                while next_i < self.frame_count:
                    if not self.df.loc[next_i, 'clean_ball_position']:
                        next_i += 1
                        continue
                    else:
                        # Clear the next position as it's causing the issue
                        self.df.loc[next_i, 'clean_ball_position'].clear()
                        print(f'Cleared ball position at frame: {next_i}')
                        break
                        
                # Skip ahead to avoid removing multiple related positions
                i = next_i + 1
            else:
                # Clear current position
                self.df.loc[i, 'clean_ball_position'].clear()
                print(f'Cleared ball position at frame: {i}')
                # Skip ahead
                i += 10
                
            removed = True
            
        return removed

    def draw(self, input_frames: List[np.ndarray], show_velocity: bool = True, 
             show_position: bool = True) -> List[np.ndarray]:
        """
        Draw bounding boxes and tracking information on input frames.
        
        Args:
            input_frames: List of video frames
            show_velocity: Whether to display velocity information
            show_position: Whether to display position information
            
        Returns:
            Frames with tracking visualization
        """
        if not self.complete_ball_positions:
            print("No ball positions available to draw")
            return input_frames
            
        if len(input_frames) != len(self.complete_ball_positions):
            print(f"Warning: Frame count mismatch. Frames: {len(input_frames)}, "
                  f"Ball positions: {len(self.complete_ball_positions)}")
            # Use the shorter list length
            frames_to_process = min(len(input_frames), len(self.complete_ball_positions))
        else:
            frames_to_process = len(input_frames)
            
        output_frames = []
        
        for i in range(frames_to_process):
            # Create a copy to avoid modifying original frames
            frame_copy = input_frames[i].copy()
            ball_position = self.complete_ball_positions[i]
            
            for track_id, bbox in ball_position.items():
                if not bbox or len(bbox) != 4:
                    continue
                    
                # Create bounding box for easier handling
                box = BoundingBox.from_list(bbox)
                
                # Draw bounding box
                cv2.rectangle(frame_copy, (box.x1, box.y1), (box.x2, box.y2), (0, 0, 255), 2)
                
                # Draw track ID
                cv2.putText(
                    frame_copy, str(track_id), (box.x1, box.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
                
                # Add additional information if available
                if self.df is not None and i < len(self.df):
                    # Add velocity information
                    if show_velocity:
                        velocity = self.df.loc[i, 'velocity']
                        cv2.putText(
                            frame_copy, f"v: {velocity:.1f}", (box.x1, box.y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                        )
                    
                    # Add position information
                    if show_position:
                        x, y = self.df.loc[i, 'x'], self.df.loc[i, 'y']
                        cv2.putText(
                            frame_copy, f"({x}, {y})", (box.x2 + 5, box.y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                        )
                        
            output_frames.append(frame_copy)
            
        return output_frames
    