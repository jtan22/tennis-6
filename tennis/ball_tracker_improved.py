"""
Ball Tracking System for Tennis Video Analysis
==============================================

This module provides functionality to track a tennis ball in video footage
and analyze its trajectory to detect hits and bounces.

Physics reference:
- Drag force: Fd = 1/2*Cd*p*A*v^2
  (Cd=0.6, p=1.225 kg/m3, A=0.0034 m2 → Fd ≈ 0.00125*v^2)
- Magnus force: Fl = Cl*p*A*v^2
  (Cl depends on spin, typically 0.2 for 30 m/s with 50 rps)
- Terminal velocity: Vt ≈ 21.84 m/s vertically
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pickle
import torch
from scipy.signal import find_peaks
from ultralytics import YOLO

@dataclass
class BoundingBox:
    """Represents a bounding box around a detected ball."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def center(self) -> Tuple[int, int]:
        """Return the center point of the bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    @property
    def width(self) -> int:
        """Return the width of the bounding box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """Return the height of the bounding box."""
        return self.y2 - self.y1
    
    def to_list(self) -> List[int]:
        """Convert to list format [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]
    
    @classmethod
    def from_list(cls, coords: List[int]) -> 'BoundingBox':
        """Create a BoundingBox from a list [x1, y1, x2, y2]."""
        if len(coords) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")
        return cls(coords[0], coords[1], coords[2], coords[3])
    
    @classmethod
    def from_center_and_size(cls, center: Tuple[int, int], width: int, height: int) -> 'BoundingBox':
        """Create a BoundingBox from center point and dimensions."""
        x, y = center
        half_width = width // 2
        half_height = height // 2
        return cls(
            x - half_width, 
            y - half_height, 
            x + half_width, 
            y + half_height
        )


class BallDetector:
    """Handles detection of the ball in individual frames."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.15):
        """Initialize the ball detector with a YOLOv8 model.
        
        Args:
            model_path: Path to the YOLOv8 model file
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Use appropriate device
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        self.model.to(device)
        print(f"Using device: {device}")
    
    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """Detect balls in a single frame.
        
        Args:
            frame: Image as numpy array
            
        Returns:
            List of BoundingBox objects for detected balls
        """
        results = self.model.predict(frame, conf=self.confidence_threshold)[0]
        detections = []
        
        for box in results.boxes:
            # Extract coordinates and convert to integers
            coords = box.xyxy.tolist()[0]
            bbox = BoundingBox(
                int(coords[0]),
                int(coords[1]),
                int(coords[2]),
                int(coords[3])
            )
            detections.append(bbox)
            
        return detections


class TrajectoryUtils:
    """Utility functions for trajectory analysis."""
    
    @staticmethod
    def distance_between_points(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    @staticmethod
    def distance_from_point_to_line(point: Tuple[int, int], 
                                    line_p1: Tuple[int, int], 
                                    line_p2: Tuple[int, int]) -> float:
        """Calculate distance from point to line segment."""
        # Line vector
        line_vec = (line_p2[0] - line_p1[0], line_p2[1] - line_p1[1])
        
        # Vector from line_p1 to point
        point_vec = (point[0] - line_p1[0], point[1] - line_p1[1])
        
        # Line length squared
        line_len_sq = line_vec[0]**2 + line_vec[1]**2
        
        # If line has zero length, return distance to line_p1
        if line_len_sq == 0:
            return np.sqrt(point_vec[0]**2 + point_vec[1]**2)
        
        # Calculate projection of point_vec onto line_vec
        t = max(0, min(1, (point_vec[0]*line_vec[0] + point_vec[1]*line_vec[1]) / line_len_sq))
        
        # Calculate closest point on line segment
        closest_x = line_p1[0] + t * line_vec[0]
        closest_y = line_p1[1] + t * line_vec[1]
        
        # Return distance to closest point
        return np.sqrt((point[0] - closest_x)**2 + (point[1] - closest_y)**2)
    
    @staticmethod
    def exponential_deceleration(start: float, end: float, steps: int, exponent: float = 1.2) -> List[float]:
        """Generate a sequence simulating exponential deceleration.
        
        Args:
            start: Starting value
            end: Ending value
            steps: Number of steps
            exponent: Exponent for the deceleration curve
            
        Returns:
            List of values representing the deceleration
        """
        if steps <= 0:
            return []
        if steps == 1:
            return [start]
            
        sequence = [
            start + (end - start) * (1 - (1 - i / (steps - 1)) ** exponent) 
            for i in range(steps)
        ]
        return sequence


class BallTracker:
    """Tracks a tennis ball throughout video frames and analyzes its trajectory."""
    
    def __init__(self, model_path: str):
        """Initialize ball tracker with a detection model.
        
        Args:
            model_path: Path to the YOLOv8 model file
        """
        self.detector = BallDetector(model_path)
        self.utils = TrajectoryUtils()
        
        # Data structures for tracking
        self.multiple_ball_positions = []  # Raw detections per frame
        self.single_ball_positions = []    # Best ball selected per frame
        self.complete_ball_positions = []  # After interpolation
        self.frame_count = 0
        
        # Analysis results
        self.near_hits = []
        self.near_bounces = []
        self.far_hits = []
        self.far_bounces = []
        self.hits_and_bounces = []
        self.near_hit_bounce_pairs = []
        self.far_hit_bounce_pairs = []
    
    def detect_ball_positions_all_frames(self, frames: List[np.ndarray], 
                                        read_from_stub: bool = False, 
                                        stub_path: Optional[str] = None) -> None:
        """Process all frames to detect ball positions.
        
        Args:
            frames: List of video frames
            read_from_stub: Whether to read from saved data
            stub_path: Path to saved data file
        """
        if read_from_stub and stub_path:
            try:
                with open(stub_path, 'rb') as f:
                    self.multiple_ball_positions = pickle.load(f)
                self.frame_count = len(self.multiple_ball_positions)
                print(f'Loaded {self.frame_count} frames of ball position data')
                return
            except (FileNotFoundError, pickle.PickleError) as e:
                print(f"Error loading from stub: {e}. Proceeding with detection.")
        
        self.multiple_ball_positions = []
        for i, frame in enumerate(frames):
            if i % 100 == 0:
                print(f"Processing frame {i}/{len(frames)}")
                
            detections = self.detector.detect(frame)
            # Convert to old format for compatibility
            frame_data = {}
            for idx, bbox in enumerate(detections, 1):
                frame_data[idx] = bbox.to_list()
            
            self.multiple_ball_positions.append(frame_data)
        
        self.frame_count = len(self.multiple_ball_positions)
        print(f'Detected ball in {self.frame_count} frames')
        
        if stub_path:
            try:
                with open(stub_path, 'wb') as f:
                    pickle.dump(self.multiple_ball_positions, f)
                print(f"Saved detection data to {stub_path}")
            except Exception as e:
                print(f"Error saving to stub: {e}")
    
    def analyse_ball_positions_all_frames(self, fps: int) -> pd.DataFrame:
        """Analyze ball positions across all frames.
        
        Args:
            fps: Frames per second of the video
            
        Returns:
            DataFrame with analysis results
        """
        # 1. Select the best ball in each frame (remove extras)
        self.single_ball_positions = self._remove_all_extra_balls_detected()
        
        # 2. Clean up and interpolate missing positions
        df = self._clean_up_ball_positions(self.single_ball_positions)
        
        # 3. Find hits and bounces
        self._find_ball_hits_and_bounces(fps, df)
        
        return df
    
    def _remove_all_extra_balls_detected(self) -> List[Dict[int, List[int]]]:
        """Select the best ball candidate in frames with multiple detections."""
        single_ball_positions = []
        
        for i, frame_data in enumerate(self.multiple_ball_positions):
            # If 0 or 1 ball, keep as is
            if len(frame_data) <= 1:
                single_ball_positions.append(frame_data)
                continue
                
            # Find last known position
            last_position = self._find_last_ball_position(i)
            next_position = self._find_next_ball_position(i)
            
            # If we can't determine trajectory, keep first ball
            if last_position is None or next_position is None:
                single_ball_positions.append({1: frame_data[1]} if frame_data else {})
                continue
            
            # Get the ball closest to the expected trajectory
            best_ball = self._find_best_ball_position(last_position, frame_data, next_position)
            single_ball_positions.append({1: best_ball})
            
        return single_ball_positions
    
    def _find_last_ball_position(self, index: int) -> Optional[List[int]]:
        """Find the last frame with a single ball detection."""
        if index == 0:
            return None
            
        for i in range(index - 1, -1, -1):
            if len(self.multiple_ball_positions[i]) == 1:
                return self.multiple_ball_positions[i][1]
                
        return None
    
    def _find_next_ball_position(self, index: int) -> Optional[List[int]]:
        """Find the next frame with a single ball detection."""
        if index >= len(self.multiple_ball_positions) - 1:
            return None
            
        for i in range(index + 1, len(self.multiple_ball_positions)):
            if len(self.multiple_ball_positions[i]) == 1:
                return self.multiple_ball_positions[i][1]
                
        return None
    
    def _find_best_ball_position(self, 
                               last_ball_position: List[int], 
                               ball_positions: Dict[int, List[int]], 
                               next_ball_position: List[int]) -> List[int]:
        """Find the most likely ball position based on trajectory."""
        # Get centers for line calculation
        line_start = self._get_center_point(last_ball_position)
        line_end = self._get_center_point(next_ball_position)
        
        min_distance = float('inf')
        best_ball_id = 0
        
        for ball_id, position in ball_positions.items():
            point = self._get_center_point(position)
            distance = self.utils.distance_from_point_to_line(point, line_start, line_end)
            
            if distance < min_distance:
                min_distance = distance
                best_ball_id = ball_id
                
        return ball_positions[best_ball_id]
    
    def _get_center_point(self, bbox: List[int]) -> Tuple[int, int]:
        """Calculate center point of a bounding box."""
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    
    def _clean_up_ball_positions(self, single_ball_positions: List[Dict[int, List[int]]]) -> pd.DataFrame:
        """Clean up ball positions and prepare data for analysis."""
        clean_positions = single_ball_positions.copy()
        
        # Collect initial statistics
        df = self._collect_stats(clean_positions)
        
        # Remove outliers based on velocity
        mean_velocity = np.mean(df['velocity'])
        std_velocity = np.std(df['velocity'])
        cutoff_velocity = mean_velocity + 2 * std_velocity
        print(f'Velocity stats: mean={mean_velocity:.2f}, std={std_velocity:.2f}, cutoff={cutoff_velocity:.2f}')
        
        # Iteratively remove invalid positions until no more are found
        iterations = 0
        while self._remove_invalid_ball_positions(cutoff_velocity, df) and iterations < 10:
            df = self._collect_stats(clean_positions)
            iterations += 1
            
        print(f"Removed outliers in {iterations} iterations")
        
        return df
    
    def _interpolate_ball_positions(self, ball_positions: List[Dict[int, List[int]]]) -> List[Dict[int, List[int]]]:
        """Fill in missing ball positions using interpolation."""
        # Extract bounding boxes
        bounding_boxes = [pos.get(1, []) for pos in ball_positions]
        
        # Find ranges with missing data
        missing_ranges = []
        start_index = None
        
        for i, bbox in enumerate(bounding_boxes):
            if not bbox:  # Empty list means missing position
                if start_index is None:
                    start_index = i
            elif start_index is not None:
                missing_ranges.append((start_index, i - 1))
                start_index = None
                
        # Still missing at the end?
        if start_index is not None:
            missing_ranges.append((start_index, len(bounding_boxes) - 1))
        
        # Skip first range if it starts at 0 (can't interpolate)
        if missing_ranges and missing_ranges[0][0] == 0:
            missing_ranges = missing_ranges[1:]
        
        # Interpolate each missing range
        for start, end in missing_ranges:
            if start == 0 or end >= len(bounding_boxes) - 1:
                continue  # Can't interpolate at boundaries
                
            # Get known positions before and after gap
            start_bbox = bounding_boxes[start - 1]
            end_bbox = bounding_boxes[end + 1]
            
            if not start_bbox or not end_bbox:
                continue  # Skip if we don't have valid boundaries
                
            start_pos = self._get_center_point(start_bbox)
            end_pos = self._get_center_point(end_bbox)
            
            # Calculate interpolation parameters
            steps = end - start + 3
            
            # Generate x and y sequences
            seq_x = self.utils.exponential_deceleration(start_pos[0], end_pos[0], steps)
            seq_y = self.utils.exponential_deceleration(start_pos[1], end_pos[1], steps)
            
            # Use only interior points
            seq_x = seq_x[1:-1]
            seq_y = seq_y[1:-1]
            
            # Calculate average size
            width = ((start_bbox[2] - start_bbox[0]) + (end_bbox[2] - end_bbox[0])) // 2
            height = ((start_bbox[3] - start_bbox[1]) + (end_bbox[3] - end_bbox[1])) // 2
            
            # Create interpolated bounding boxes
            for i, (x, y) in enumerate(zip(seq_x, seq_y)):
                bounding_boxes[start + i] = [
                    int(x - width / 2),
                    int(y - height / 2),
                    int(x + width / 2),
                    int(y + height / 2)
                ]
        
        # Create DataFrame for final interpolation of any remaining gaps
        df = pd.DataFrame(bounding_boxes, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate().bfill().ffill()
        
        # Convert back to list of dictionaries
        complete_positions = [{1: row.tolist()} for _, row in df.iterrows()]
        
        return complete_positions
    
    def _collect_stats(self, clean_ball_positions: List[Dict[int, List[int]]]) -> pd.DataFrame:
        """Collect statistics from ball positions for analysis."""
        # Interpolate missing positions
        self.complete_ball_positions = self._interpolate_ball_positions(clean_ball_positions)
        
        # Create base DataFrame
        df = pd.DataFrame({
            'multi_ball_position': self.multiple_ball_positions,
            'clean_ball_position': clean_ball_positions,
            'ball_position': self.complete_ball_positions,
        })
        
        # Extract center points
        df['center_point'] = df['ball_position'].apply(
            lambda pos: self._get_center_point(pos[1]) if 1 in pos else None
        )
        
        # Split into x, y columns
        centers = pd.DataFrame(df['center_point'].tolist(), columns=['x', 'y'], index=df.index)
        df = pd.concat([df, centers], axis=1)
        
        # Calculate motion metrics
        df['x_diff'] = df['x'].diff().fillna(0).astype(int)
        df['y_diff'] = df['y'].diff().fillna(0).astype(int)
        df['frame'] = df.index
        
        # Velocity calculations
        df['velocity'] = np.sqrt(df['x_diff']**2 + df['y_diff']**2).round(2)
        
        # Use signed velocity (up is negative, down is positive)
        df['velocity_vector'] = np.where(
            df['y_diff'] >= 0,
            df['velocity'],
            -df['velocity']
        )
        
        # Calculate rolling metrics
        window_size = 5
        df['y_rolling_mean'] = df['y'].rolling(window=window_size, min_periods=1).mean().round(1)
        df['y_rolling_mean_delta'] = df['y_rolling_mean'].diff().round(1)
        df['velocity_vector_delta'] = df['velocity_vector'].diff().round(2)
        df['velocity_rolling_mean'] = df['velocity'].rolling(window=window_size, min_periods=1).mean().round().astype(int)
        
        # Save for debugging
        df.to_csv('ball_data_frame.csv', index=False)
        
        return df
    
    def _remove_invalid_ball_positions(self, cutoff_velocity: float, df: pd.DataFrame) -> bool:
        """Remove ball positions with impossible velocity values."""
        removed = False
        i = 1
        
        while i < self.frame_count:
            if df.loc[i, 'velocity'] < cutoff_velocity:
                i += 1
                continue
                
            # Found outlier
            print(f"Found outlier at frame {i} with velocity {df.loc[i, 'velocity']}")
            
            if not df.loc[i, 'clean_ball_position']:
                # Outlier caused by next position - find and remove it
                i += 1
                while i < self.frame_count:
                    if not df.loc[i, 'clean_ball_position']:
                        i += 1
                    else:
                        df.loc[i, 'clean_ball_position'].clear()
                        print(f"Cleared ball position at frame {i}")
                        break
            else:
                # Outlier is this position
                df.loc[i, 'clean_ball_position'].clear()
                print(f"Cleared ball position at frame {i}")
                
            # Skip ahead to avoid cascading effects
            i += 10
            removed = True
            
        return removed
    
    def _find_ball_hits_and_bounces(self, fps: int, df: pd.DataFrame) -> None:
        """Identify ball hits and bounces throughout the video."""
        # Find peaks in velocity (potential bounces)
        peaks, _ = find_peaks(df['velocity_rolling_mean'], width=5, prominence=5)
        troughs, _ = find_peaks(-df['velocity_rolling_mean'], prominence=5)
        
        print(f"Found {len(peaks)} velocity peaks and {len(troughs)} troughs")
        
        # Identify different types of events
        self.near_hits = self._find_near_hits(df)
        self.near_bounces = self._find_near_bounces(peaks, fps, df)
        self.far_bounces = self._find_far_bounces(troughs, fps, df)
        self.far_hits = self._find_far_hits(df)
        
        # Organize hits and bounces into sequence
        self.hits_and_bounces = self._sort_hits_and_bounces()
        self.near_hit_bounce_pairs = self._find_near_hit_bounce_pairs()
        self.far_hit_bounce_pairs = self._find_far_hit_bounce_pairs()
    
    def _find_near_hits(self, df: pd.DataFrame) -> List[int]:
        """Find frames where the near player hits the ball."""
        # Use statistical threshold to identify sudden changes
        mean = np.mean(df['velocity_vector_delta'])
        std = np.std(df['velocity_vector_delta'])
        cutoff = mean - 3 * std
        
        print(f'Velocity delta stats: mean={mean:.2f}, std={std:.2f}, cutoff={cutoff:.2f}')
        
        # Find candidates with significant velocity changes
        candidates = df[df['velocity_vector_delta'] < cutoff].index.tolist()
        
        # Filter candidates by checking for consecutive negative velocity
        near_hits = []
        for candidate in candidates:
            if self._all_negative_velocity(candidate, 10, df):
                # Adjust index to match the actual hit moment
                near_hits.append(candidate - 1)
                
        print(f'Found {len(near_hits)} near hits: {near_hits}')
        return near_hits
    
    def _all_negative_velocity(self, start_index: int, count: int, df: pd.DataFrame) -> bool:
        """Check if velocity remains negative for given number of frames."""
        end_index = min(start_index + count, len(df))
        
        for i in range(start_index, end_index):
            if df.loc[i, 'velocity_vector'] >= 0:
                return False
                
        return True
    
    def _find_near_bounces(self, peaks: np.ndarray, fps: int, df: pd.DataFrame) -> List[int]:
        """Find frames where the ball bounces near the player."""
        # Filter peaks that occur near hits
        candidates = []
        for peak in peaks:
            if self._peak_in_range(peak, fps):
                candidates.append(int(peak))
                
        print(f'Found {len(candidates)} near bounce candidates')
        
        # Refine bounce detection
        near_bounces = []
        for candidate in candidates:
            sum_velocity_delta, near_bounce = self._refine_near_bounce(candidate, df)
            
            if sum_velocity_delta <= 0 and near_bounce not in near_bounces:
                near_bounces.append(near_bounce - 1)  # Adjust to actual bounce frame
                
        print(f'Found {len(near_bounces)} near bounces: {near_bounces}')
        return near_bounces
    
    def _peak_in_range(self, peak: int, fps: int) -> bool:
        """Check if peak is within appropriate time range of a near hit."""
        if not self.near_hits:
            return False
            
        # Check if peak is within 1 second before any near hit
        for near_hit in self.near_hits:
            if peak > near_hit - fps and peak < near_hit:
                return True
                
        # Or after the last near hit
        return peak > self.near_hits[-1]
    
    def _refine_near_bounce(self, candidate: int, df: pd.DataFrame) -> Tuple[float, int]:
        """Find the exact bounce frame near a velocity peak."""
        start_index = candidate
        end_index = min(start_index + 10, self._get_next_near_hit(start_index))
        
        # Look for frame with most negative velocity change
        min_velocity_delta_index = start_index
        sum_velocity_delta = 0
        
        for i in range(start_index + 1, end_index):
            delta = df.loc[i, 'velocity_vector_delta']
            if delta < df.loc[min_velocity_delta_index, 'velocity_vector_delta']:
                min_velocity_delta_index = i
                
            sum_velocity_delta += delta
            
        return sum_velocity_delta, min_velocity_delta_index
    
    def _get_next_near_hit(self, index: int) -> int:
        """Find the next near hit after the given index."""
        for near_hit in self.near_hits:
            if index < near_hit:
                return near_hit
                
        return self.frame_count - 1
    
    def _find_far_bounces(self, troughs: np.ndarray, fps: int, df: pd.DataFrame) -> List[int]:
        """Find frames where the ball bounces far from the player."""
        # Determine valid frame ranges for far bounces
        frame_ranges = self._find_far_bounce_frame_ranges(fps)
        
        # Find candidates within valid ranges
        candidates = []
        for trough in troughs:
            if self._trough_in_range(trough, frame_ranges):
                candidates.append(int(trough))
                
        print(f'Found {len(candidates)} far bounce candidates')
        
        # Refine far bounce detection
        far_bounces = []
        for candidate in candidates:
            far_bounce = self._refine_far_bounce(candidate, df)
            if far_bounce not in far_bounces:
                far_bounces.append(far_bounce - 1)  # Adjust to actual bounce frame
                
        print(f'Found {len(far_bounces)} far bounces: {far_bounces}')
        return far_bounces
    
    def _find_far_bounce_frame_ranges(self, fps: int) -> List[Tuple[int, int]]:
        """Calculate valid frame ranges where far bounces might occur."""
        after_hit_buffer = fps // 2      # 0.5 seconds after hit
        before_bounce_buffer = fps       # 1 second before bounce
        
        # Combine and sort all near events
        near_events = sorted(self.near_hits + self.near_bounces)
        
        frame_ranges = []
        i = 0
        
        while i < len(near_events):
            # Find next hit
            hit_index = self._find_next_near_hit_index(i, near_events)
            if hit_index >= len(near_events):
                break
                
            i = hit_index + 1
            
            # Find next bounce
            bounce_index = self._find_next_near_bounce_index(i, near_events)
            
            if bounce_index >= len(near_events):
                # No more bounces, use remaining frames
                range_start = int(near_events[hit_index] + after_hit_buffer)
                frame_ranges.append((range_start, self.frame_count))
                break
            else:
                # Add range between hit and bounce
                range_start = int(near_events[hit_index] + after_hit_buffer)
                range_end = int(near_events[bounce_index] - before_bounce_buffer)
                
                if range_end > range_start:
                    frame_ranges.append((range_start, range_end))
                    
                i = bounce_index + 1
                
        return frame_ranges
    
    def _find_next_near_hit_index(self, start_index: int, events: List[int]) -> int:
        """Find the next near hit in the combined event list."""
        for i in range(start_index, len(events)):
            if events[i] in self.near_hits:
                return i
                
        return len(events)
    
    # def _find_next_near_bounce_index(self, start_index: int, events: List[int]) -> int:
    #     """Find the next near bounce in the combined event list."""
    #     for i in range(start_index, len(events)):
    #         if events[]

def find_next_near_bounce_index(self, start_index: int, events: List[int]) -> int:
    """
    Find the next index in the events list that corresponds to a near bounce.
    
    Args:
        start_index: The starting index to search from
        events: A sorted list containing both near hits and near bounces
        
    Returns:
        The index of the next near bounce, or the length of events if none found
    """
    for i in range(start_index, len(events)):
        if events[i] in self.near_bounces:
            return i
    return len(events)

def trough_in_range(self, trough: int, frame_ranges: List[Tuple[int, int]]) -> bool:
    """
    Check if the given trough falls within any of the specified frame ranges.
    
    Args:
        trough: Frame number to check
        frame_ranges: List of tuples with (start_frame, end_frame)
        
    Returns:
        True if the trough is within any range, False otherwise
    """
    for start_frame, end_frame in frame_ranges:
        if start_frame < trough < end_frame:
            return True
    return False

def refine_far_bounce(self, candidate: int, df: pd.DataFrame) -> int:
    """
    Find the exact frame of a far bounce by identifying the frame with the 
    lowest velocity delta near the candidate frame.
    
    Args:
        candidate: The candidate frame for a far bounce
        df: DataFrame containing velocity data
        
    Returns:
        The refined frame number for the far bounce
    """
    start_index = candidate
    end_index = min(start_index + 10, self.frame_count)
    
    # Find frame with minimum velocity delta
    min_velocity_delta_index = start_index
    for i in range(start_index + 1, end_index):
        if df.loc[i, 'velocity_vector_delta'] < df.loc[min_velocity_delta_index, 'velocity_vector_delta']:
            min_velocity_delta_index = i
            
    return min_velocity_delta_index

def find_far_hits(self, df: pd.DataFrame) -> List[int]:
    """
    Identify frames where the far player hits the ball by analyzing velocity vectors
    and filtering out near hits.
    
    Args:
        df: DataFrame containing ball tracking data
        
    Returns:
        List of frame indices representing far hits
    """
    # Find all potential ball hits
    self.ball_hits = self.find_ball_hits(df)
    logger.debug(f'All detected hits: {self.ball_hits}')
    
    # Filter out hits that are too close to near hits
    candidates = [hit for hit in self.ball_hits if not self.hit_in_near_hits_range(hit)]
    logger.debug(f'Far hits candidates: {candidates}')
    
    # Refine each candidate to find the exact hit frame
    far_hits = []
    for candidate in candidates:
        far_hit = self.refine_far_hit(candidate, df)
        if far_hit is not None:
            far_hits.append(far_hit)
    
    # Adjust indices to match the actual hit frames
    far_hits = [item - 1 for item in far_hits]
    logger.debug(f'Refined far hits: {far_hits}')
    
    return far_hits

def refine_far_hit(self, candidate: int, df: pd.DataFrame) -> Optional[int]:
    """
    Refine the far hit index by finding the transition point from negative to positive
    velocity vectors followed by a consistent positive trend.
    
    Args:
        candidate: The candidate frame for a far hit
        df: DataFrame containing velocity data
        
    Returns:
        The refined frame number for the far hit, or None if invalid
    """
    # Find the transition point from negative to positive velocity
    start_index = candidate
    while start_index > 0 and df.loc[start_index, 'velocity_vector'] >= 0:
        start_index -= 1
    start_index += 1
    
    # Verify that the velocity remains positive for the next 10 frames
    # to confirm this is a genuine hit and not noise
    consecutive_positive_frames = 0
    for i in range(start_index, min(start_index + 10, len(df))):
        if df.loc[i, 'velocity_vector'] >= 0:
            consecutive_positive_frames += 1
        else:
            break
            
    if consecutive_positive_frames >= 8:  # Allow for small fluctuations
        return start_index
    return None

def hit_in_near_hits_range(self, hit: int, buffer: int = 10) -> bool:
    """
    Check if a potential hit is close to an already identified near hit.
    
    Args:
        hit: Frame number to check
        buffer: Number of frames before and after a near hit to consider "close"
        
    Returns:
        True if the hit is within the buffer range of any near hit
    """
    return any(near_hit - buffer <= hit <= near_hit + buffer for near_hit in self.near_hits)

def find_ball_hits(self, df: pd.DataFrame) -> List[int]:
    """
    Identify potential ball hits by analyzing changes in the y-axis movement direction.
    
    Args:
        df: DataFrame containing ball tracking data
        
    Returns:
        List of frame indices representing potential hits
    """
    # Initialize the hit detection column
    df['ball_hit'] = 0
    minimum_consecutive_frames = 25
    window_size = int(minimum_consecutive_frames * 1.2)
    
    for i in range(1, len(df) - window_size):
        # Detect changes in direction of vertical movement
        direction_change = (
            (df.at[i, 'y_rolling_mean_delta'] > 0 and df.at[i+1, 'y_rolling_mean_delta'] < 0) or
            (df.at[i, 'y_rolling_mean_delta'] < 0 and df.at[i+1, 'y_rolling_mean_delta'] > 0)
        )
        
        if direction_change:
            # Determine the new direction
            new_direction_positive = df.at[i+1, 'y_rolling_mean_delta'] > 0
            
            # Count consistent frames in the new direction
            consistent_frames = 0
            for j in range(i + 1, i + window_size + 1):
                if (new_direction_positive and df.at[j, 'y_rolling_mean_delta'] > 0) or \
                   (not new_direction_positive and df.at[j, 'y_rolling_mean_delta'] < 0):
                    consistent_frames += 1
            
            # Mark as a hit if we have enough consistent frames
            if consistent_frames >= minimum_consecutive_frames:
                df.loc[i, 'ball_hit'] = 1
    
    return df[df['ball_hit'] == 1].index.tolist()

def sort_hits_and_bounces(self) -> List[Optional[int]]:
    """
    Organize detected hits and bounces into the expected sequence:
    far-hit -> near-bounce -> near-hit -> far-bounce -> far-hit -> ...
    
    Returns:
        List of frame indices (or None for missing events) in sequential order
    """
    # Combine all events into a sorted array
    all_events = np.concatenate((self.near_hits, self.far_hits, self.near_bounces, self.far_bounces))
    all_events = np.sort(all_events)
    logger.debug(f'All hits and bounces (sorted): {all_events}')
    
    # Create the sequence
    sequence = []
    event_types = ['far_hit', 'near_bounce', 'near_hit', 'far_bounce']
    current_type_index = 0
    i = 0
    
    while True:
        # Determine current event type we're looking for
        current_type = event_types[current_type_index]
        event_array = getattr(self, current_type + 's')  # e.g., self.far_hits
        
        # Find matching event
        found_event = None
        while i < len(all_events):
            if all_events[i] in event_array:
                found_event = int(all_events[i])
                i += 1
                break
            i += 1
            
        # Add event to sequence (or None if not found)
        sequence.append(found_event)
        
        # Move to next event type
        current_type_index = (current_type_index + 1) % len(event_types)
        
        # Break if we've processed all events
        if i >= len(all_events) and found_event is None:
            break
            
    logger.debug(f'Sequenced hits and bounces: {sequence}')
    return sequence

def find_near_hit_bounce_pairs(self) -> List[Tuple[int, int]]:
    """
    Extract pairs of near hits and their corresponding far bounces.
    
    Returns:
        List of (near_hit, far_bounce) frame index tuples
    """
    pairs = []
    for i in range(2, len(self.hits_and_bounces) - 1, 4):
        hit = self.hits_and_bounces[i]
        bounce = self.hits_and_bounces[i + 1]
        if hit is not None and bounce is not None:
            pairs.append((hit, bounce))
    
    logger.debug(f'Near hit-bounce pairs: {pairs}')
    return pairs

def find_far_hit_bounce_pairs(self) -> List[Tuple[int, int]]:
    """
    Extract pairs of far hits and their corresponding near bounces.
    
    Returns:
        List of (far_hit, near_bounce) frame index tuples
    """
    pairs = []
    for i in range(0, len(self.hits_and_bounces) - 1, 4):
        hit = self.hits_and_bounces[i]
        bounce = self.hits_and_bounces[i + 1]
        if hit is not None and bounce is not None:
            pairs.append((hit, bounce))
    
    logger.debug(f'Far hit-bounce pairs: {pairs}')
    return pairs

def draw(self, input_frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Draw ball tracking bounding boxes and event markers on each frame.
    
    Args:
        input_frames: List of video frames
        
    Returns:
        List of frames with ball tracking and event annotations
    """
    output_frames = []
    for frame_idx, (frame, ball_position) in enumerate(zip(input_frames, self.complete_ball_positions)):
        # Create a copy to avoid modifying the original frame
        annotated_frame = frame.copy()
        
        # Draw ball position
        for track_id, bounding_box in ball_position.items():
            x1, y1, x2, y2 = [int(coord) for coord in bounding_box]
            
            # Draw bounding box and ID
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated_frame, str(track_id), 
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
            
            # Mark special events on frames
            self._mark_events(annotated_frame, frame_idx)
            
        output_frames.append(annotated_frame)
    return output_frames

def _mark_events(self, frame: np.ndarray, frame_idx: int) -> None:
    """
    Mark special events (hits and bounces) on the frame.
    
    Args:
        frame: Frame to annotate
        frame_idx: Index of the current frame
    """
    # Define colors and labels for different event types
    event_markers = [
        (self.near_hits, (0, 255, 0), "Near Hit"),
        (self.far_hits, (255, 0, 0), "Far Hit"),
        (self.near_bounces, (0, 255, 255), "Near Bounce"),
        (self.far_bounces, (255, 255, 0), "Far Bounce")
    ]
    
    for events, color, label in event_markers:
        if frame_idx in events:
            # Draw a colored circle and label at the top of the frame
            height, width = frame.shape[:2]
            cv2.circle(frame, (width // 2, 50), 25, color, -1)
            cv2.putText(
                frame, label, 
                (width // 2 - 60, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )