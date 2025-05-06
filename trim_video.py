import cv2
import logging
import time
import numpy as np
from tennis import court_line_detector
from tennis.court_line_detector import CourtLineDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    INPUT_VIDEO_PATH                = 'output_videos/sample-whole-01.mp4'
    OUTPUT_VIDEO_PATH               = 'output_videos/sample-whole-01.avi'
    COURT_LINE_DETECTOR_MODEL_PATH  = 'models/keypoints/keypoints_model_resnet50_epoch20_cuda.pth'

    logger.info('Start processing...')

    video_in = cv2.VideoCapture(INPUT_VIDEO_PATH)
    fps = round(video_in.get(cv2.CAP_PROP_FPS))
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # type: ignore
    frame_size = (width, height)
    video_out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, frame_size)

    court_line_detector = CourtLineDetector(COURT_LINE_DETECTOR_MODEL_PATH)
    start_time_mono = time.monotonic()
    i = 0
    while True:
        ret, frame = video_in.read()
        if not ret:
            break
        process_frame(i, frame, court_line_detector)
        cv2.putText(frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        video_out.write(frame)
        i += 1
        if i % 100 == 0:
            print(f'Processed {i} frames, took [{(time.monotonic() - start_time_mono):.2f}] seconds')
    video_in.release()
    video_out.release()

    logger.info('Finished processing') 

def process_frame(frame_number: int, frame: np.ndarray, court_line_detector: CourtLineDetector) -> None:
    try:
        court_line_detector.detect_keypoints(frame)
    except ValueError:
        print(f'{frame_number} not a court')

    refined_predicted_keypoints_count = sum(1 for keypoint in court_line_detector.refined_predicted_keypoints if keypoint is not None)
    if refined_predicted_keypoints_count >= 3:
        cv2.putText(frame, f'Tennis Court {refined_predicted_keypoints_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f'{refined_predicted_keypoints_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    court_line_detector.draw_all_keypoints(frame)

if __name__ == "__main__":
    main()

