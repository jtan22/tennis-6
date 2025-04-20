import cv2
from shapely.geometry import Point, LineString
import math
from .constants import GRAVITY, BALL_TERMINAL_VELOCITY_SQUARED

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
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
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
def get_distance_by_time(initial_velocity, time):
    return (BALL_TERMINAL_VELOCITY_SQUARED / GRAVITY) * math.log((BALL_TERMINAL_VELOCITY_SQUARED + (GRAVITY * initial_velocity * time)) / BALL_TERMINAL_VELOCITY_SQUARED)

