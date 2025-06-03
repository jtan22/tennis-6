import cv2
import logging
import time
from networkx import jaccard_coefficient
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Tuple
from tennis.court_line_detector import CourtLineDetector
from tennis.player_tracker import PlayerTracker
from tennis.ball_tracker import BallTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_VIDEO_PATH                = 'output_videos/sample-whole-01.mp4'
OUTPUT_VIDEO_PATH               = 'output_videos/sample-whole-01.avi'
COURT_LINE_DETECTOR_MODEL_PATH  = 'models/keypoints/keypoints_model_resnet50_epoch20_cuda.pth'
KEY_DATA_PATH                   = 'analysis/key_data.csv'
PLAYER_TRACKER_MODEL_PATH       = 'yolov8x'
BALL_TRACKER_MODEL_PATH         = 'models/ball/train-5l6u-10-64/weights/last-378-213.pt'
FPS = 25

def main():

    COLLECT_KEY_DATA = False
    ANALYSE_KEY_DTAT = True

    if COLLECT_KEY_DATA:
        collect_key_data()

    if ANALYSE_KEY_DTAT:
        analyse_key_data()

def analyse_key_data():
    df = pd.read_csv(KEY_DATA_PATH)
    keypoints_counts = df['keypoints_count'].to_list()
    person_positions = [eval(x) for x in df['person_positions'].to_list()]
    ball_positions = [eval(x) for x in df['ball_positions'].to_list()]
    court_ranges = find_frame_ranges_with_court(keypoints_counts)
    print(f'court_ranges: {court_ranges}')

def find_frame_ranges_with_court(keypoints_counts: List[int]) -> List[Tuple[int, int]]:
    court_ranges = []
    last_10: deque[bool] = deque([False] * 10, maxlen=10)
    in_range = False
    range_start = 0
    range_end = 0
    for i in range(len(keypoints_counts)):
        court_detected = keypoints_counts[i] > 2
        last_10.append(court_detected)
        if in_range and last_10.count(False) >= 8:
            first_false_index = next(j for j, value in enumerate(last_10) if not value)
            range_end = i - 10 + first_false_index
            court_ranges.append((range_start, range_end))
            in_range = False
        elif not in_range and last_10.count(True) >= 8:
            first_true_index = next(j for j, value in enumerate(last_10) if value)
            range_start = i - 10 + first_true_index + 1
            in_range = True
    if in_range:
        court_ranges.append((range_start, len(keypoints_counts) - 1))


    return court_ranges

def collect_key_data():
    logger.info('Start collecting key data...')

    video_in = cv2.VideoCapture(INPUT_VIDEO_PATH)
    fps = round(video_in.get(cv2.CAP_PROP_FPS))
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'H264') # type: ignore
    frame_size = (width, height)
    video_out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, frame_size)

    court_line_detector = CourtLineDetector(COURT_LINE_DETECTOR_MODEL_PATH)
    player_tracker = PlayerTracker(PLAYER_TRACKER_MODEL_PATH)
    ball_tracker = BallTracker(BALL_TRACKER_MODEL_PATH)
    key_data = []
    start_time_mono = time.monotonic()
    i = 0
    while True:
        ret, frame = video_in.read()
        if not ret:
            break

        try:
            court_line_detector.detect_keypoints(frame)
        except ValueError:
            # print(f'{frame_number} not a court')
            pass
        person_positions = player_tracker.detect_person_positions_per_frame(frame)
        ball_positions = ball_tracker.detect_ball_positions_per_frame(frame)
        
        key_datum = {
            'frame_number': i,
            'keypoints_count': sum(1 for keypoint in court_line_detector.refined_predicted_keypoints if keypoint is not None),
            'person_positions': person_positions,
            'ball_positions': ball_positions,
        }
        key_data.append(key_datum)

        cv2.putText(frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        court_line_detector.draw_all_keypoints(frame)
        player_tracker.draw_positions(frame, person_positions)
        ball_tracker.draw_positions(frame, ball_positions)

        video_out.write(frame)
        i += 1
        if i % 100 == 0:
            print(f'Processed {i} frames, took [{(time.monotonic() - start_time_mono):.2f}] seconds')
    video_in.release()
    video_out.release()

    pd.DataFrame(key_data).to_csv(KEY_DATA_PATH, index=False)
    logger.info('Finished collecting key data') 

if __name__ == "__main__":
    main()

