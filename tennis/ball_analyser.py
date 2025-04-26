from scipy.signal import find_peaks
import numpy as np
from typing import List, Tuple, Optional, Any
import pandas as pd

class BallAnalyser:
    """
    Analyzes ball movement data to identify hits and bounces in a ping pong/tennis match.
    
    Detects near hits (player closest to camera), far hits (player further from camera),
    near bounces (bounces on near side), and far bounces (bounces on far side).
    """

    def __init__(self, frame_count: int):
        """
        Initialize the BallAnalyser.
        
        Args:
            frame_count: Total number of frames in the video.
        """
        self.frame_count = frame_count
        
        # Results storage
        self.hits_and_bounces: Optional[List[Optional[int]]] = None
        self.ball_hits: Optional[List[int]] = None

    def find_ball_hits_and_bounces(self, fps: int, df: pd.DataFrame) -> None:
        """
        Main method to identify all ball hits and bounces in the data.
        
        Args:
            fps: Frames per second of the video.
            df: DataFrame containing ball tracking data with columns:
                - velocity_rolling_mean
                - velocity_vector
                - velocity_vector_delta
                - y_rolling_mean_delta
        """
        # Signed velocity (positive when moving down, negative when moving up)
        df['velocity_vector'] = np.where(
            df['y_diff'] >= 0,
            df['velocity'],
            -df['velocity']
        ).astype(float)
        df['velocity_vector_delta'] = df['velocity_vector'].diff().round(2)
        df['velocity_vector_delta'] = df['velocity_vector_delta'].astype(float)
        
        # Calculate smoothed metrics
        window_size = min(5, len(df))
        if window_size > 0:
            rolling_args = {'window': window_size, 'min_periods': 1, 'center': False}
            
            df['y_rolling_mean'] = df['y'].rolling(**rolling_args).mean().round(1)
            df['y_rolling_mean_delta'] = df['y_rolling_mean'].diff().round(1)
            df['velocity_rolling_mean'] = df['velocity'].rolling(**rolling_args).mean().round(1)
            
            # Acceleration (second derivative)
            df['acceleration'] = df['velocity_vector'].diff().rolling(**rolling_args).mean().round(2)

        # Find velocity peaks and troughs
        peaks, _ = find_peaks(df['velocity_rolling_mean'], width=5, prominence=5)
        troughs, _ = find_peaks(-df['velocity_rolling_mean'], prominence=5)
        
        print(f'Original peaks: {peaks}')
        print(f'Original troughs: {troughs}')

        # Identify different types of events
        near_hits = self._find_near_hits(df)
        near_bounces = self._find_near_bounces(peaks, fps, df, near_hits)
        far_bounces = self._find_far_bounces(troughs, fps, df, near_hits, near_bounces)
        far_hits = self._find_far_hits(df, near_hits)
        
        # Process event relationships
        self.hits_and_bounces = self._sort_hits_and_bounces(near_hits, near_bounces, far_hits, far_bounces)

    def _find_near_hits(self, df: Any) -> List[int]:
        """
        Find near hits (hits by player closest to camera) by identifying 
        large negative velocity deltas followed by consistent negative velocity.
        
        Args:
            df: DataFrame with velocity data.
            
        Returns:
            List of frame indices representing near hits.
        """
        # Use statistical approach to find cutoff for significant velocity changes
        mean = np.mean(df['velocity_vector_delta'])
        std = np.std(df['velocity_vector_delta'])
        cutoff = mean - 3 * std
        print(f'Velocity delta: mean {mean}, std {std}, cutoff {cutoff}')
        
        # Find candidate frames with significant velocity changes
        candidates = [i for i in range(2, self.frame_count) 
                     if df.loc[i, 'velocity_vector_delta'] < cutoff]
        print(f'Near hits candidates: {candidates}')
        
        # Filter candidates to ensure they have consistent negative velocity afterward
        near_hits = [candidate - 1 for candidate in candidates 
                    if self._all_negative_velocity(candidate, 10, df)]
        print(f'Near hits: {near_hits}')
        
        return near_hits

    def _all_negative_velocity(self, start_index: int, count: int, df: Any) -> bool:
        """
        Check if velocity vectors are negative for a consecutive number of frames.
        
        Args:
            start_index: First frame to check.
            count: Number of consecutive frames to check.
            df: DataFrame with velocity data.
            
        Returns:
            True if all frames have negative velocity, False otherwise.
        """
        end_index = min(start_index + count, self.frame_count)
        return all(df.loc[i, 'velocity_vector'] < 0 for i in range(start_index, end_index))

    def _find_near_bounces(self, peaks: np.ndarray, fps: int, df: pd.DataFrame, near_hits: List[int]) -> List[int]:
        """
        Find near bounces (bounces on the near side of the court).
        
        Args:
            peaks: Velocity peak indices.
            fps: Frames per second.
            df: DataFrame with velocity data.
            
        Returns:
            List of frame indices representing near bounces.
        """
        print(f'frame_range: {fps}')
        
        # Filter peaks to those within range of a near hit
        candidates = [int(peak) for peak in peaks if self._peak_in_range(peak, fps, near_hits)]
        print(f'Near bounce candidates: {candidates}')
        
        # Refine bounce detection
        near_bounces = []
        for candidate in candidates:
            sum_velocity_delta, near_bounce = self._refine_near_bounce(candidate, df, near_hits)
            if sum_velocity_delta <= 0 and near_bounce not in near_bounces:
                near_bounces.append(near_bounce)
                
        near_bounces = [item - 1 for item in near_bounces]
        print(f'Near bounces: {near_bounces}')
        
        return near_bounces

    def _peak_in_range(self, peak: int, frame_range: int, near_hits: List[int]) -> bool:
        """
        Check if a peak is within the expected range before a near hit.
        
        Args:
            peak: Frame index of the peak.
            frame_range: Number of frames to consider (typically fps).
            
        Returns:
            True if the peak is in range, False otherwise.
        """
        # Check if peak is within frame_range frames before any near hit
        for near_hit in near_hits:
            if near_hit - frame_range < peak < near_hit:
                return True
                
        # Also accept peaks after the last near hit
        return peak > near_hits[-1] if near_hits else False

    def _refine_near_bounce(self, candidate: int, df: pd.DataFrame, near_hits: List[int]) -> Tuple[float, int]:
        """
        Refine near bounce detection by finding the frame with lowest velocity delta.
        
        Args:
            candidate: Candidate frame index.
            df: DataFrame with velocity data.
            
        Returns:
            Tuple of (sum of velocity deltas, refined bounce frame index).
        """
        start_index = candidate
        end_index = min(start_index + 10, self._get_next_near_hit(start_index, near_hits))
        min_velocity_delta_index = start_index
        sum_velocity_delta: float = 0
        
        for i in range(start_index + 1, end_index):
            df['velocity_vector_delta'] = df['velocity_vector_delta'].astype(float)
            if df.loc[i, 'velocity_vector_delta'] < df.loc[min_velocity_delta_index, 'velocity_vector_delta']:
                min_velocity_delta_index = i
            sum_velocity_delta += df.loc[i, 'velocity_vector_delta']
            
        return sum_velocity_delta, min_velocity_delta_index

    def _get_next_near_hit(self, index: int, near_hits: List[int]) -> int:
        """
        Get the frame index of the next near hit after a given index.
        
        Args:
            index: Current frame index.
            
        Returns:
            Frame index of next near hit or last frame if none found.
        """
        for near_hit in near_hits:
            if index < near_hit:
                return near_hit
        return self.frame_count - 1

    def _find_far_bounces(self, troughs: np.ndarray, fps: int, df: pd.DataFrame, near_hits: List[int], near_bounces: List[int]) -> List[int]:
        """
        Find far bounces (bounces on the far side of the court).
        
        Args:
            troughs: Velocity trough indices.
            fps: Frames per second.
            df: DataFrame with velocity data.
            
        Returns:
            List of frame indices representing far bounces.
        """
        # Find valid frame ranges where far bounces could occur
        frame_ranges = self._find_far_bounce_frame_ranges(fps, near_hits, near_bounces)
        print(f'frame ranges: {frame_ranges}')
        
        # Filter troughs to those within valid ranges
        candidates = [int(trough) for trough in troughs 
                     if self._trough_in_range(trough, frame_ranges)]
        print(f'Far bounce candidates: {candidates}')
        
        # Refine bounce detection
        far_bounces = []
        for candidate in candidates:
            far_bounce = self._refine_far_bounce(candidate, df)
            if far_bounce not in far_bounces:
                far_bounces.append(far_bounce)
                
        far_bounces = [item - 1 for item in far_bounces]
        print(f'Far bounces: {far_bounces}')
        
        return far_bounces

    def _find_far_bounce_frame_ranges(self, fps: int, near_hits: List[int], near_bounces: List[int]) -> List[Tuple[int, int]]:
        """
        Find potential frame ranges where far bounces could occur.
        Typically between near hits and near bounces with appropriate buffers.
        
        Args:
            fps: Frames per second.
            
        Returns:
            List of tuples (start_frame, end_frame) for valid far bounce ranges.
        """
        after_hit_buffer = fps / 2
        before_bounce_buffer = fps
        frame_ranges = []
        
        # Combine and sort all near hits and bounces
        near_hits_bounces = np.sort(np.concatenate((near_hits, near_bounces)))
        print(f'near hits bounces: {near_hits_bounces}')
        
        i = 0
        while i < len(near_hits_bounces):
            # Find next hit-bounce sequence
            hit_index = self._find_next_near_hit_index(i, near_hits_bounces, near_hits)
            if hit_index == len(near_hits_bounces):
                break
                
            i = hit_index + 1
            bounce_index = self._find_next_near_bounce_index(i, near_hits_bounces, near_bounces)
            
            # Define range between hit and bounce (or to end of frames)
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
    
    def _find_next_near_hit_index(self, index: int, near_hits_bounces: np.ndarray, near_hits: List[int]) -> int:
        """
        Find the next index in near_hits_bounces array that corresponds to a near hit.
        
        Args:
            index: Starting index to search from.
            near_hits_bounces: Combined array of near hits and bounces.
            
        Returns:
            Index of next near hit or length of array if none found.
        """
        for i in range(index, len(near_hits_bounces)):
            if near_hits_bounces[i] in near_hits:
                return i
        return len(near_hits_bounces)

    def _find_next_near_bounce_index(self, index: int, near_hits_bounces: np.ndarray, near_bounces: List[int]) -> int:
        """
        Find the next index in near_hits_bounces array that corresponds to a near bounce.
        
        Args:
            index: Starting index to search from.
            near_hits_bounces: Combined array of near hits and bounces.
            
        Returns:
            Index of next near bounce or length of array if none found.
        """
        for i in range(index, len(near_hits_bounces)):
            if near_hits_bounces[i] in near_bounces:
                return i
        return len(near_hits_bounces)
    
    def _trough_in_range(self, trough: int, frame_ranges: List[Tuple[int, int]]) -> bool:
        """
        Check if a trough is within any valid frame range.
        
        Args:
            trough: Frame index of the trough.
            frame_ranges: List of valid frame ranges (start_frame, end_frame).
            
        Returns:
            True if trough is in a valid range, False otherwise.
        """
        return any(start < trough < end for start, end in frame_ranges)

    def _refine_far_bounce(self, candidate: int, df: Any) -> int:
        """
        Refine far bounce detection by finding frame with lowest velocity delta.
        
        Args:
            candidate: Candidate frame index.
            df: DataFrame with velocity data.
            
        Returns:
            Refined bounce frame index.
        """
        start_index = candidate
        end_index = min(start_index + 10, self.frame_count)
        
        # Find frame with lowest velocity delta
        min_velocity_delta_index = start_index
        for i in range(start_index + 1, end_index):
            if df.loc[i, 'velocity_vector_delta'] < df.loc[min_velocity_delta_index, 'velocity_vector_delta']:
                min_velocity_delta_index = i
                
        return min_velocity_delta_index

    def _find_far_hits(self, df: pd.DataFrame, near_hits: List[int]) -> List[int]:
        """
        Find far hits (hits by player furthest from camera).
        
        Args:
            df: DataFrame with velocity data.
            
        Returns:
            List of frame indices representing far hits.
        """
        # Identify all ball hits first
        self.ball_hits = self._find_ball_hits(df)
        print(f'All hits: {self.ball_hits}')
        
        # Filter out hits that are too close to near hits
        candidates = [hit for hit in self.ball_hits 
                     if not self._hit_in_near_hits_range(hit, near_hits)]
        print(f'Far hits candidates: {candidates}')
        
        # Refine far hit detection
        far_hits = []
        for candidate in candidates:
            far_hit = self._refine_far_hit(candidate, df)
            if far_hit is not None and far_hit not in far_hits:
                far_hits.append(far_hit)
                
        far_hits = [item - 1 for item in far_hits]
        print(f'Far hits: {far_hits}')
        
        return far_hits
    
    def _refine_far_hit(self, candidate: int, df: Any) -> Optional[int]:
        """
        Refine far hit detection by finding velocity vector direction change.
        
        Args:
            candidate: Candidate frame index.
            df: DataFrame with velocity data.
            
        Returns:
            Refined hit frame index or None if not a valid hit.
        """
        # Find the transition from negative to positive velocity
        start_index = candidate
        while start_index > 0 and df.loc[start_index, 'velocity_vector'] >= 0:
            start_index -= 1
        start_index += 1
        
        # Verify that velocity remains positive after the hit
        for i in range(start_index, min(start_index + 10, self.frame_count)):
            if df.loc[i, 'velocity_vector'] < 0:
                return None
                
        return start_index

    def _hit_in_near_hits_range(self, hit: int, near_hits: List[int]) -> bool:
        """
        Check if a hit is too close to any near hit.
        
        Args:
            hit: Frame index to check.
            
        Returns:
            True if hit is within range of a near hit, False otherwise.
        """
        return any(near_hit - 10 < hit < near_hit + 10 for near_hit in near_hits)

    def _find_ball_hits(self, df: Any) -> List[int]:
        """
        Find all potential ball hits by analyzing y-axis movement direction changes.
        
        Args:
            df: DataFrame with ball tracking data.
            
        Returns:
            List of frame indices representing potential ball hits.
        """
        df = df.copy()  # Avoid modifying the original dataframe
        df['ball_hit'] = 0
        minimum_change_frames_for_hit = 25
        check_frames = int(minimum_change_frames_for_hit * 1.2)
        
        for i in range(1, len(df) - check_frames):
            # Detect direction changes in y-axis movement
            negative_change = df.at[i, 'y_rolling_mean_delta'] > 0 and df.at[i+1, 'y_rolling_mean_delta'] < 0
            positive_change = df.at[i, 'y_rolling_mean_delta'] < 0 and df.at[i+1, 'y_rolling_mean_delta'] > 0
            
            if negative_change or positive_change:
                change_count = 0
                
                # Count how many following frames have the same directional change
                for j in range(i + 1, i + check_frames + 1):
                    still_negative = df.at[i, 'y_rolling_mean_delta'] > 0 and df.at[j, 'y_rolling_mean_delta'] < 0
                    still_positive = df.at[i, 'y_rolling_mean_delta'] < 0 and df.at[j, 'y_rolling_mean_delta'] > 0
                    
                    if (negative_change and still_negative) or (positive_change and still_positive):
                        change_count += 1
                        
                # Mark as hit if sufficient consecutive frames show consistent change
                if change_count >= minimum_change_frames_for_hit:
                    df.loc[i, 'ball_hit'] = 1
                    
        return df[df['ball_hit'] == 1].index.tolist()

    def _sort_hits_and_bounces(self, 
                               near_hits: List[int], 
                               near_bounces: List[int], 
                               far_hits: List[int], 
                               far_bounces: List[int]) -> List[Optional[int]]:
        """
        Sort all hits and bounces into expected sequence pattern:
        far-hit → near-bounce → near-hit → far-bounce
        
        Returns:
            List of frame indices in sequence, with None for missing events.
        """
        # Combine all events and sort by frame number
        candidates = np.sort(np.concatenate((
            near_hits, 
            far_hits, 
            near_bounces, 
            far_bounces
        )))
        print(f'Sort hits and bounces: {candidates}')
        
        hits_and_bounces = []
        event_types = ['far_hit', 'near_bounce', 'near_hit', 'far_bounce']
        current_type_index = 0
        i = 0
        
        # Assign each event to its expected position in sequence
        while i < len(candidates):
            event_frame = candidates[i]
            current_type = event_types[current_type_index]
            
            # Check if current frame matches expected event type
            if (current_type == 'far_hit' and event_frame in far_hits) or \
               (current_type == 'near_bounce' and event_frame in near_bounces) or \
               (current_type == 'near_hit' and event_frame in near_hits) or \
               (current_type == 'far_bounce' and event_frame in far_bounces):
                hits_and_bounces.append(int(event_frame))
                i += 1
            else:
                hits_and_bounces.append(None)
                
            # Move to next event type in cycle
            current_type_index = (current_type_index + 1) % 4
            
        print(f'Sorted hits and bounces: {hits_and_bounces}')
        return hits_and_bounces    
    