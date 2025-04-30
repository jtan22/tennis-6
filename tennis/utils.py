import cv2
from shapely.geometry import Point, LineString
import math
from collections import deque
from .constants import GRAVITY, BALL_TERMINAL_VELOCITY_SQUARED, DEFAULT_VERTICAL_VELOCITY, \
    AIR_DENSITY, BALL_DRAG_COEFFICIENT, BALL_CROSS_SECTION, BALL_MASS, BALL_DRAG_FACTOR, \
    DEFAULT_PLAYER_HEIGHT, PLAYER_STANDING_WIDTH_HEIGHT_RATIO

def read_video(video_path):
    # Read a video file and return its frames as a list of numpy arrays
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
    video_capture.release()
    return frames, fps

def save_video(video_frames, fps, video_path):
    # Get 4-character code for MJPG codec
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # type: ignore
    frame_size = (video_frames[0].shape[1], video_frames[0].shape[0])
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    for frame in video_frames:
        out.write(frame)
    out.release()

def get_bounding_box_center_point(bounding_box):
    x1, y1, x2, y2 = bounding_box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (int(center_x), int(center_y))

def get_bottom_line_center_point(bounding_box):
    x1, y1, x2, y2 = bounding_box
    center_x = (x1 + x2) / 2
    return (int(center_x), int(y2))

def get_distance_between_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance

def get_distance_between_point_and_line(point, line_start, line_end):
    point_sh = Point(point[0], point[1])
    line_sh = LineString([line_start, line_end])
    distance_infinite_sh = point_sh.distance(line_sh)
    print(f"Shapely: Distance from {point_sh} to the line {line_sh}: {distance_infinite_sh}")
    return distance_infinite_sh

# U0 = (Vt^2)*(e^(g*x/Vt^2) - 1)/(g*t)
def get_initial_horizontal_velocity(distance, time):
    return BALL_TERMINAL_VELOCITY_SQUARED * (math.e ** (distance * GRAVITY / BALL_TERMINAL_VELOCITY_SQUARED) - 1) / (GRAVITY * time)

# x = (Vt^2/g)*ln((Vt^2+g*U0*t)/Vt^2) = (Vt^2/g)*ln(1+g*U0*t/Vt^2)
def get_horizontal_distance_by_time(initial_velocity, time):
    return (BALL_TERMINAL_VELOCITY_SQUARED / GRAVITY) * math.log((BALL_TERMINAL_VELOCITY_SQUARED + (GRAVITY * initial_velocity * time)) / BALL_TERMINAL_VELOCITY_SQUARED)

# u = (Vt^2*U0)/(Vt^2+g*U0*t)
def get_horizontal_velocity_by_time(initial_velocity, time):
    return (BALL_TERMINAL_VELOCITY_SQUARED * initial_velocity) / (BALL_TERMINAL_VELOCITY_SQUARED + GRAVITY * initial_velocity * time)

# Upward acceleration:   -g - (rho * Cd * A * v**2) / (2 * m)
# Downward acceleration: -g + (rho * Cd * A * v**2) / (2 * m) 
def simulate_vertical_motion_hit(v0, h_initial, rho, Cd, A, m, g):
    dt = 0.001  # Time step
    t = 0
    y = h_initial
    v = v0
    
    # Upward motion
    while v > 0:
        v = v + dt * (-g - (rho * Cd * A * v**2) / (2 * m))
        y = y + v * dt
        t += dt
    
    # At the top, velocity becomes 0
    v = 0
    
    # Downward motion
    while y > 0:
        v = v + dt * (-g + (rho * Cd * A * v**2) / (2 * m))
        y = y + v * dt
        t += dt
    
    return t, y

def get_initial_vertical_velocity_hit(initial_height, total_seconds):
    print(f'Initial height: {initial_height}, Total seconds: {total_seconds}')
    # Initial guess for v0
    v0_guess = DEFAULT_VERTICAL_VELOCITY

    # Iterative search for v0
    tolerance = 0.001
    max_iterations = 1000
    last_2_v0 = deque(maxlen=2)
    last_2_tf = deque(maxlen=2)
    for i in range(max_iterations):
        time_of_flight, height_reached = simulate_vertical_motion_hit(v0_guess, 
                                                  initial_height, 
                                                  AIR_DENSITY, 
                                                  BALL_DRAG_COEFFICIENT, 
                                                  BALL_CROSS_SECTION, 
                                                  BALL_MASS, 
                                                  GRAVITY)
        print(f'iteration: {i}, time_of_flight: {time_of_flight}, height_reached: {height_reached}, vo_guess: {v0_guess}')

        if v0_guess in last_2_v0:
            v = last_2_v0[0] - last_2_v0[1]
            t = last_2_tf[0] - last_2_tf[1]
            td = last_2_tf[0] - total_seconds
            v0_guess = last_2_v0[0] - (td / t) * v
            break
        else:
            last_2_v0.append(v0_guess)
            last_2_tf.append(time_of_flight)

        if abs(time_of_flight - total_seconds) < tolerance:
            break
        elif time_of_flight < total_seconds:
            v0_guess += 0.1  # Increase v0
        else:
            v0_guess -= 0.1  # Decrease v0
    
    print(f'Final v0_guess: {v0_guess}')
    return v0_guess

# Upward acceleration:   -g - (rho * Cd * A * v**2) / (2 * m)
# Downward acceleration: -g + (rho * Cd * A * v**2) / (2 * m) 
def simulate_vertical_motion_bounce(v0, total_seconds, rho, Cd, A, m, g):
    dt = 0.001  # Time step
    t = 0
    y = 0
    v = v0
    
    # Upward motion
    while t < total_seconds and v > 0:
        v = v + dt * (-g - (rho * Cd * A * v**2) / (2 * m))
        y = y + v * dt
        t += dt
    
    # Downward motion
    while t < total_seconds and v <= 0:
        v = v + dt * (-g + (rho * Cd * A * v**2) / (2 * m))
        y = y + v * dt
        t += dt
    
    return y

def get_initial_vertical_velocity_bounce(final_height, total_seconds):
    print(f'Final height: {final_height}, Total seconds: {total_seconds}')
    # Initial guess for v0
    v0_guess = DEFAULT_VERTICAL_VELOCITY

    # Iterative search for v0
    tolerance = 0.01
    max_iterations = 1000
    last_2_v0 = deque(maxlen=2)
    last_2_hr = deque(maxlen=2)
    for i in range(max_iterations):
        height_reached = simulate_vertical_motion_bounce(v0_guess, 
                                                  total_seconds,
                                                  AIR_DENSITY, 
                                                  BALL_DRAG_COEFFICIENT, 
                                                  BALL_CROSS_SECTION, 
                                                  BALL_MASS, 
                                                  GRAVITY)
        print(f'iteration: {i}, height_reached: {height_reached}, vo_guess: {v0_guess}')

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
        elif height_reached < final_height:
            v0_guess += 0.1  # Increase v0
        else:
            v0_guess -= 0.1  # Decrease v0
    
    print(f'Final v0_guess: {v0_guess}')
    return v0_guess

def get_vertical_distances_and_velocities(v0, h_initial, dt, t_max=5.0):
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

def get_hypotenuse(a, b):
    return round((a ** 2 + b **2) ** 0.5, 2)

def get_player_height(player_ratio):
    return DEFAULT_PLAYER_HEIGHT * PLAYER_STANDING_WIDTH_HEIGHT_RATIO ** 0.1 / player_ratio ** 0.1
