import cv2
import math
import numpy as np
import sympy
import logging
from shapely.geometry import Point, LineString
from sympy import Line
from scipy.spatial import distance
from typing import List, Tuple, Optional
from collections import deque
from .constants import (
    GRAVITY, 
    BALL_TERMINAL_VELOCITY_SQUARED, 
    DEFAULT_VERTICAL_VELOCITY,
    BALL_MASS,
    BALL_DRAG_FACTOR,
    COURT_HOMOGRAPHY_CONFIGURATIONS,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_video(video_path: str) -> Tuple[List[np.ndarray], int]:
    # Read a video file and return its frames as a list of numpy arrays
    video_capture = cv2.VideoCapture(video_path)
    fps = round(video_capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
    video_capture.release()
    return frames, fps

def save_video(video_frames: List[np.ndarray], fps: int, video_path: str) -> None:
    # Get 4-character code for MJPG codec
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # type: ignore
    frame_size = (video_frames[0].shape[1], video_frames[0].shape[0])
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    for frame in video_frames:
        out.write(frame)
    out.release()

def get_bottom_line_center_point(bounding_box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bounding_box
    center_x = (x1 + x2) / 2
    return (int(center_x), int(y2))

def get_distance_between_points(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    x1, y1 = point1
    x2, y2 = point2
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance

def get_distance_between_point_and_line(point: Tuple[int, int], line_start: Tuple[int, int], line_end: Tuple[int, int]) -> float:
    point_sh = Point(point[0], point[1])
    line_sh = LineString([line_start, line_end])
    distance_infinite_sh = point_sh.distance(line_sh)
    return distance_infinite_sh

def line_intersection(line1: np.ndarray, line2: np.ndarray) -> Optional[Tuple[int, int]]:
    try:
        # Create SymPy Line objects
        l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
        l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))
        
        # Find intersection
        intersection = l1.intersection(l2) # type: ignore
        
        # Process result
        if not intersection or len(intersection) == 0:
            return None
        
        if isinstance(intersection[0], sympy.geometry.point.Point2D): # type: ignore
            x, y = intersection[0].coordinates
            return (int(x), int(y))
        else:
            return None
    except Exception as e:
        logger.warning(f"Error calculating line intersection: {e}")

def merge_lines(lines: List[np.ndarray]) -> List[np.ndarray]:
    if lines is None or len(lines) == 0:
        return []
        
    # Sort lines based on x-coordinate
    lines = sorted(lines, key=lambda line: line[0])
    
    # Track which lines have been merged
    mask = [True] * len(lines)
    merged_lines = []
    
    # Merge similar lines
    for i, line in enumerate(lines):
        if not mask[i]:
            continue

        curr_line = line.copy()
        
        # Check subsequent lines for possible merges
        for j in range(i + 1, len(lines)):
            if not mask[j]:
                continue

            # Extract line coordinates
            x1, y1, x2, y2 = curr_line
            x3, y3, x4, y4 = lines[j]
            
            # Calculate distances between endpoints
            dist1 = distance.euclidean((x1, y1), (x3, y3))
            dist2 = distance.euclidean((x2, y2), (x4, y4))
            
            # If endpoints are close, merge the lines
            if dist1 < 20 and dist2 < 20:
                curr_line = np.array([
                    int((x1 + x3) / 2), 
                    int((y1 + y3) / 2),
                    int((x2 + x4) / 2), 
                    int((y2 + y4) / 2)
                ], dtype=np.int32)
                
                # Mark the merged line
                mask[j] = False
        
        merged_lines.append(curr_line)
            
    return merged_lines

def get_homography_matrix(standard_keypoints: List[Tuple[int, int]], image_keypoints: List[Tuple[int, int]]) -> np.ndarray:
    homography_matrix = None
    distance_min = np.inf
    # Turn 2D keypoints into 3D, (14, 2) to (14, 1, 2)
    reference_keypoints = np.array(standard_keypoints, dtype=np.float32).reshape((-1, 1, 2))
    
    # Try each homography configuration to find the best one
    for homography_configuration in COURT_HOMOGRAPHY_CONFIGURATIONS:
        # The 4 standard keypoints
        standard_configuration = [standard_keypoints[i] for i in homography_configuration]
        # The 4 image keypoints
        image_configuration = [image_keypoints[i] for i in homography_configuration]
        # Skip if any keypoint is missing
        if any(point is None for point in image_configuration):
            continue
        # Calculate homography matrix
        matrix, _ = cv2.findHomography(
            np.float32(standard_configuration), np.float32(image_configuration), method=cv2.RANSAC) # type: ignore        
        # Apply homography to all reference keypoints
        transformed_keypoints = cv2.perspectiveTransform(reference_keypoints, matrix)
        # Calculate error for non-used keypoints
        distances = []
        for i in range(len(standard_keypoints)):
            if i not in homography_configuration and image_keypoints[i] is not None:
                distances.append(distance.euclidean(image_keypoints[i], transformed_keypoints[i][0]))
        
        # If no distances to compare, skip this configuration
        if not distances:
            continue
            
        distance_mean = np.mean(distances)
        if distance_mean < distance_min:
            homography_matrix = matrix
            distance_min = distance_mean
    
    if homography_matrix is None:
        raise ValueError("Could not find valid homography matrix with given keypoints")

    return homography_matrix        

def transform_coordinates(coordinates: List[Tuple[int, int]], homography_matrix: np.ndarray) -> List[Tuple[int, int]]:
    coordinates_3d = np.array(coordinates, dtype=np.float32).reshape((-1, 1, 2))
    transformed_coordinates = cv2.perspectiveTransform(coordinates_3d, homography_matrix)
    return [(int(x), int(y)) for x, y in transformed_coordinates.reshape(-1, 2)]

# U0 = (Vt^2)*(e^(g*x/Vt^2) - 1)/(g*t)
def get_initial_horizontal_velocity(distance: float, time: float) -> float:
    v0 = BALL_TERMINAL_VELOCITY_SQUARED * (math.e ** (distance * GRAVITY / BALL_TERMINAL_VELOCITY_SQUARED) - 1) / (GRAVITY * time)
    return v0

# x = (Vt^2/g)*ln((Vt^2+g*U0*t)/Vt^2) = (Vt^2/g)*ln(1+g*U0*t/Vt^2)
def get_horizontal_distance_by_time(initial_velocity: float, time: float) -> float:
    return (BALL_TERMINAL_VELOCITY_SQUARED / GRAVITY) * math.log((BALL_TERMINAL_VELOCITY_SQUARED + (GRAVITY * initial_velocity * time)) / BALL_TERMINAL_VELOCITY_SQUARED)

# u = (Vt^2*U0)/(Vt^2+g*U0*t)
def get_horizontal_velocity_by_time(initial_velocity: float, time: float) -> float:
    return (BALL_TERMINAL_VELOCITY_SQUARED * initial_velocity) / (BALL_TERMINAL_VELOCITY_SQUARED + GRAVITY * initial_velocity * time)

def get_initial_vertical_velocity_hit(initial_height: float, total_seconds: float) -> float:
    # Initial guess for v0
    v0_guess = DEFAULT_VERTICAL_VELOCITY

    # Iterative search for v0
    tolerance = 0.001
    max_iterations = 100
    last_2_v0: deque[float] = deque(maxlen=2)
    last_2_tf: deque[float] = deque(maxlen=2)
    for i in range(max_iterations):
        time_of_flight = simulate_vertical_motion_hit(v0_guess, initial_height)

        if v0_guess in last_2_v0:
            v = last_2_v0[0] - last_2_v0[1]
            t = last_2_tf[0] - last_2_tf[1]
            td = last_2_tf[0] - total_seconds
            v0_guess = last_2_v0[0] - (td / t) * v
            break
        elif time_of_flight in last_2_tf:
            break            
        else:
            last_2_v0.append(v0_guess)
            last_2_tf.append(time_of_flight)

        if abs(time_of_flight - total_seconds) < tolerance:
            break

        adjustment = calculate_time_adjustment(last_2_tf, time_of_flight, total_seconds)
        if time_of_flight < total_seconds:
            v0_guess += adjustment  # Increase v0
        else:
            v0_guess -= adjustment  # Decrease v0
    return v0_guess

# Upward acceleration:   -g - (rho * Cd * A * v**2) / (2 * m)
# Downward acceleration: -g + (rho * Cd * A * v**2) / (2 * m) 
def simulate_vertical_motion_hit(v0: float, h_initial: float) -> float:
    dt = 0.001  # Time step
    t = 0
    y = h_initial
    v = v0
    
    # Upward motion
    while v > 0:
        v = v + dt * (-GRAVITY - (BALL_DRAG_FACTOR * v**2) / (2 * BALL_MASS))
        y = y + v * dt
        t += dt
    
    # Downward motion
    while y > 0:
        v = v + dt * (-GRAVITY + (BALL_DRAG_FACTOR * v**2) / (2 * BALL_MASS))
        y = y + v * dt
        t += dt
    
    return t

def calculate_time_adjustment(last_2_tf: deque[float], time_of_flight: float, total_seconds: float) -> float:
    if len(last_2_tf) < 2:
        return 0.1
    delta = abs(last_2_tf[0] - last_2_tf[1])
    diff = abs(time_of_flight - total_seconds)
    magnitude = diff / delta
    if magnitude > 5:
        return 0.1 * magnitude
    else:
        return 0.1

def get_initial_vertical_velocity_bounce(final_height: float, total_seconds: float) -> float:
    # Initial guess for v0
    v0_guess = DEFAULT_VERTICAL_VELOCITY

    # Iterative search for v0
    tolerance = 0.01
    max_iterations = 1000
    last_2_v0: deque[float] = deque(maxlen=2)
    last_2_hr: deque[float] = deque(maxlen=2)
    for i in range(max_iterations):
        height_reached = simulate_vertical_motion_bounce(v0_guess, total_seconds)

        if v0_guess in last_2_v0:
            v = last_2_v0[0] - last_2_v0[1]
            h = last_2_hr[0] - last_2_hr[1]
            hd = last_2_hr[0] - final_height
            v0_guess = last_2_v0[0] - (hd / h) * v
            break
        else:
            last_2_v0.append(v0_guess)
            last_2_hr.append(height_reached)

        if abs(height_reached - final_height) < tolerance:
            break

        adjustment = calculate_height_adjustment(last_2_hr, height_reached, final_height)
        if height_reached < final_height:
            v0_guess += adjustment  # Increase v0
        else:
            v0_guess -= adjustment  # Decrease v0
    return v0_guess

# Upward acceleration:   -g - (rho * Cd * A * v**2) / (2 * m)
# Downward acceleration: -g + (rho * Cd * A * v**2) / (2 * m) 
def simulate_vertical_motion_bounce(v0: float, total_seconds: float) -> float:
    dt = 0.001  # Time step
    t = 0
    y = 0
    v = v0
    
    # Upward motion
    while t < total_seconds and v > 0:
        v = v + dt * (-GRAVITY - (BALL_DRAG_FACTOR * v**2) / (2 * BALL_MASS))
        y = y + v * dt
        t += dt
    
    # Downward motion
    while t < total_seconds and v <= 0:
        v = v + dt * (-GRAVITY + (BALL_DRAG_FACTOR * v**2) / (2 * BALL_MASS))
        y = y + v * dt
        t += dt
    
    return y

def calculate_height_adjustment(last_2_hr: deque[float], height_reached: float, final_height: float) -> float:
    if len(last_2_hr) < 2:
        return 0.1
    delta = abs(last_2_hr[0] - last_2_hr[1])
    diff = abs(height_reached - final_height)
    magnitude = diff / delta
    if magnitude > 5:
        return 0.1 * magnitude
    else:
        return 0.1

def get_vertical_distances_and_velocities(v0: float, h_initial: float, dt: float, t_max: float=5.0) -> Tuple[List[float], List[float]]:
    """Simulates the trajectory of the object with air resistance."""
    t = 0
    y = h_initial
    v = v0
    
    distances = [y]
    velocities = [v]
    
    # Upward motion
    while v > 0 or y > 0:  # Continue until it returns to or below initial height
        if v >= 0:
            v = v + dt * (-GRAVITY - (BALL_DRAG_FACTOR * v**2) / (2 * BALL_MASS))
        else:
            v = v + dt * (-GRAVITY + (BALL_DRAG_FACTOR * v**2) / (2 * BALL_MASS))
            
        y = y + v * dt
        t += dt
        
        distances.append(round(y, 2))
        velocities.append(round(v, 2))
        
        if t > t_max: # Safety break to prevent infinite loops
            break
            
    return distances, velocities

def get_hypotenuse(a: float, b: float) -> float:
    return round((a ** 2 + b **2) ** 0.5, 2)
