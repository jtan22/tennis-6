from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import torch
import numpy as np
from scipy.signal import find_peaks
from .utils import get_bounding_box_center_point, get_distance_between_point_and_line

class BallTracker:
    def __init__(self, model_path: str):
        """
        Initialize the BallTracker with a YOLO model.

        Args:
            model_path (str): Path to the YOLO model.
        """
        self.model = YOLO(model_path)
        self.device = torch.device(
            'mps' if torch.backends.mps.is_available() else 
            'cuda' if torch.cuda.is_available() else 
            'cpu'
        )
        self.model.to(self.device)
        self.multiple_ball_positions = []
        self.complete_ball_positions = []
        self.frame_count = 0

    def detect_ball_positions_all_frames(self, frames: list[np.ndarray], read_from_stub: bool = False, stub_path: str = None) -> None:
        """
        Detect ball positions in all frames or load from a stub file.

        Args:
            frames (list[np.ndarray]): List of video frames.
            read_from_stub (bool): Whether to read positions from a stub file.
            stub_path (str): Path to the stub file.
        """
        if read_from_stub and stub_path:
            with open(stub_path, 'rb') as f:
                self.multiple_ball_positions = pickle.load(f)
            self.frame_count = len(self.multiple_ball_positions)
            print(f'Loaded multiple ball positions size: {self.frame_count}')
            return

        self.multiple_ball_positions = [self.detect_ball_positions_per_frame(frame) for frame in frames]
        self.frame_count = len(self.multiple_ball_positions)
        print(f'Detected multiple ball positions size: {self.frame_count}')

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(self.multiple_ball_positions, f)

    def detect_ball_positions_per_frame(self, frame: np.ndarray) -> dict[int, list[int]]:
        """
        Detect ball positions in a single frame.

        Args:
            frame (np.ndarray): The input frame.

        Returns:
            dict[int, list[int]]: A dictionary with track IDs as keys and bounding boxes as values.
        """
        results = self.model.predict(frame, conf=0.15)[0]
        return {
            track_id: [int(coord) for coord in box.xyxy.tolist()[0]]
            for track_id, box in enumerate(results.boxes, start=1)
        }

    def analyse_ball_positions_all_frames(self, fps: int) -> None:
        """
        Analyze ball positions across all frames.

        Args:
            fps (int): Frames per second of the video.
        """
        single_ball_positions = self.remove_all_extra_balls_detected()
        df = self.clean_up_ball_positions(single_ball_positions)
        self.find_ball_hits_and_bounces(fps, df)

    def remove_all_extra_balls_detected(self) -> list[dict[int, list[int]]]:
        """
        Remove extra detected balls and keep only the best ball position.

        Returns:
            list[dict[int, list[int]]]: Cleaned ball positions.
        """
        single_ball_positions = []
        for i, ball_positions in enumerate(self.multiple_ball_positions):
            if len(ball_positions) < 2:
                single_ball_positions.append(ball_positions)
                continue
            last_ball_position = self.find_last_ball_position(i)
            next_ball_position = self.find_next_ball_position(i)
            if last_ball_position and next_ball_position:
                best_ball_position = self.find_best_ball_position(last_ball_position, ball_positions, next_ball_position)
                single_ball_positions.append({1: best_ball_position})
            else:
                single_ball_positions.append({1: ball_positions[1]})
        print(f'Single ball positions size: {len(single_ball_positions)}')
        return single_ball_positions

    def find_last_ball_position(self, index: int) -> list[int] | None:
        """
        Find the last available ball position with only one ball.

        Args:
            index (int): Current frame index.

        Returns:
            list[int] | None: The last ball position or None if not found.
        """
        for i in range(index - 1, -1, -1):
            if len(self.multiple_ball_positions[i]) == 1:
                return self.multiple_ball_positions[i][1]
        return None

    def find_next_ball_position(self, index: int) -> list[int] | None:
        """
        Find the next available ball position with only one ball.

        Args:
            index (int): Current frame index.

        Returns:
            list[int] | None: The next ball position or None if not found.
        """
        for i in range(index + 1, self.frame_count):
            if len(self.multiple_ball_positions[i]) == 1:
                return self.multiple_ball_positions[i][1]
        return None

    def find_best_ball_position(self, last_ball_position: list[int], ball_positions: dict[int, list[int]], next_ball_position: list[int]) -> list[int]:
        """
        Find the best ball position by minimizing the distance to the line formed by the last and next ball positions.

        Args:
            last_ball_position (list[int]): Bounding box of the last ball position.
            ball_positions (dict[int, list[int]]): Current ball positions with track IDs as keys.
            next_ball_position (list[int]): Bounding box of the next ball position.

        Returns:
            list[int]: The bounding box of the best ball position.
        """
        line_start = get_bounding_box_center_point(last_ball_position)
        line_end = get_bounding_box_center_point(next_ball_position)
        return min(
            ball_positions.values(),
            key=lambda pos: get_distance_between_point_and_line(
                get_bounding_box_center_point(pos), line_start, line_end
            )
        )

    def clean_up_ball_positions(self, single_ball_positions: list[dict[int, list[int]]]) -> pd.DataFrame:
        """
        Clean up ball positions by removing invalid positions and interpolating missing ones.

        Args:
            single_ball_positions (list[dict[int, list[int]]]): Single ball positions.

        Returns:
            pd.DataFrame: DataFrame containing cleaned ball positions.
        """
        clean_ball_positions = single_ball_positions.copy()
        df = self.collect_stats(clean_ball_positions)
        mean_velocity = df['velocity'].mean()
        std_velocity = df['velocity'].std()
        cutoff_velocity = mean_velocity + 2 * std_velocity
        print(f'Velocity: mean {mean_velocity}, std {std_velocity}, cutoff {cutoff_velocity}')
        while self.remove_invalid_ball_positions(cutoff_velocity, df):
            df = self.collect_stats(clean_ball_positions)
        self.remove_false_hits(df)
        return df

    def collect_stats(self, clean_ball_positions: list[dict[int, list[int]]]) -> pd.DataFrame:
        """
        Collect statistics for ball positions.

        Args:
            clean_ball_positions (list[dict[int, list[int]]]): Cleaned ball positions.

        Returns:
            pd.DataFrame: DataFrame containing ball position statistics.
        """
        self.complete_ball_positions = self.interpolate_ball_positions(clean_ball_positions)
        df = pd.DataFrame({
            'multi_ball_position': self.multiple_ball_positions,
            'clean_ball_position': clean_ball_positions,
            'ball_position': self.complete_ball_positions,
        })
        df['centre_point'] = df['ball_position'].apply(lambda pos: get_bounding_box_center_point(pos[1]))
        df[['x', 'y']] = pd.DataFrame(df['centre_point'].tolist(), index=df.index)
        df['x_diff'] = df['x'].diff().fillna(0).astype(int)
        df['y_diff'] = df['y'].diff().fillna(0).astype(int)
        df['velocity'] = np.sqrt(df['x_diff']**2 + df['y_diff']**2).round(2)
        return df

    def interpolate_ball_positions(self, ball_positions: list[dict[int, list[int]]]) -> list[dict[int, list[int]]]:
        """
        Interpolate missing ball positions using linear interpolation.

        Args:
            ball_positions (list[dict[int, list[int]]]): Ball positions.

        Returns:
            list[dict[int, list[int]]]: Interpolated ball positions.
        """
        bounding_boxes = [pos.get(1, []) for pos in ball_positions]
        df = pd.DataFrame(bounding_boxes, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate().bfill().ffill()
        return [{1: box.tolist()} for box in df.to_numpy()]