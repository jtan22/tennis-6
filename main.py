from tennis.ball_analyser import BallAnalyser
from tennis.ball_tracker import BallTracker
from tennis.court_line_detector import CourtLineDetector
from tennis.player_tracker import PlayerTracker
from tennis.reference_court import ReferenceCourt
from tennis.player_stats import PlayerStats
from tennis.utils import read_video, save_video
import cv2
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():

    INPUT_VIDEO_PATH                = 'input_videos/sample001.mp4'
    OUTPUT_VIDEO_PATH               = 'output_videos/output_video.avi'
    PLAYER_TRACKER_MODEL_PATH       = 'yolov8x'
    BALL_TRACKER_MODEL_PATH         = 'models/ball/train-5l6u-10-64/weights/last-378-213.pt'
    COURT_LINE_DETECTOR_MODEL_PATH  = 'models/keypoints/keypoints_model_resnet50_epoch20_cuda.pth'

    logger.info('Start processing')

    input_frames, fps = read_video(INPUT_VIDEO_PATH)
    fps = round(fps)

    DETECT_PERSON_POSITIONS         = False
    DETECT_BALL_POSITIONS           = False
    SYNTHESIZE_BALL_POSITIONS       = False
    FIND_BALL_HITS_AND_BOUNCES      = False
    DETECT_KEYPOINTS                = True
    FIND_PLAYER_POSITIONS           = True
    COMPUTE_REFERENCE_COORDINATES   = True
    COLLECT_PLAYER_STATS            = True
    DRAW_FRAMES                     = True

    if DETECT_PERSON_POSITIONS:
        logger.info('Detecting person positions...')
        start_time_mono = time.monotonic()
        PlayerTracker().dectect_person_positions(input_frames, PLAYER_TRACKER_MODEL_PATH)
        logger.info(f'Detected person positions in [{(time.monotonic() - start_time_mono):.2f}] seconds') 

    if DETECT_BALL_POSITIONS:
        logger.info('Detecting ball positions...')
        start_time_mono = time.monotonic()
        BallTracker().detect_ball_positions(input_frames, BALL_TRACKER_MODEL_PATH)
        logger.info(f'Detected person positions in [{(time.monotonic() - start_time_mono):.2f}] seconds') 

    if SYNTHESIZE_BALL_POSITIONS:
        logger.info('Synthesizing ball positions...')
        start_time_mono = time.monotonic()
        BallTracker().synthesize_ball_positions()
        logger.info(f'Detected ball positions in [{(time.monotonic() - start_time_mono):.2f}] seconds') 

    if FIND_BALL_HITS_AND_BOUNCES:
        logger.info('Finding ball hits and bounces...')
        start_time_mono = time.monotonic()
        BallAnalyser().find_ball_hits_and_bounces(fps, BallTracker().load_ball_positions_df())
        logger.info(f'Found ball hits and bounces in [{(time.monotonic() - start_time_mono):.2f}] seconds') 

    if DETECT_KEYPOINTS:
        logger.info('Detecting keypoints...')
        start_time_mono = time.monotonic()
        CourtLineDetector(COURT_LINE_DETECTOR_MODEL_PATH).detect_keypoints(input_frames[0])
        logger.info(f'Detected keypoints in [{(time.monotonic() - start_time_mono):.2f}] seconds') 

    if FIND_PLAYER_POSITIONS:
        logger.info('Finding player positions...')
        start_time_mono = time.monotonic()
        PlayerTracker().find_player_positions(CourtLineDetector(COURT_LINE_DETECTOR_MODEL_PATH).load_court_keypoints())
        logger.info(f'Found player positions in [{(time.monotonic() - start_time_mono):.2f}] seconds') 

    if COMPUTE_REFERENCE_COORDINATES:
        logger.info('Computing reference coordinates...')
        start_time_mono = time.monotonic()
        ReferenceCourt().compute_reference_coordinates(
            CourtLineDetector(COURT_LINE_DETECTOR_MODEL_PATH).load_court_keypoints(),
            *PlayerTracker().load_player_positions(), 
            BallTracker().load_complete_ball_positions(), 
            BallAnalyser().load_ball_hits_and_bounces(),
            fps)
        logger.info(f'Computed reference coordinates in [{(time.monotonic() - start_time_mono):.2f}] seconds') 
    
    if COLLECT_PLAYER_STATS:
        logger.info('Collecting player stats...')
        start_time_mono = time.monotonic()
        PlayerStats().collect_stats(ReferenceCourt().load_reference_frames())
        logger.info(f'Collected player stats in [{(time.monotonic() - start_time_mono):.2f}] seconds') 
    
    if DRAW_FRAMES:
        logger.info('Drawing player, ball, keypoints, reference court and player stats on the video...')
        start_time_mono = time.monotonic()
        output_frames = PlayerTracker().draw(input_frames)
        output_frames = BallTracker().draw(output_frames)
        output_frames = CourtLineDetector(COURT_LINE_DETECTOR_MODEL_PATH).draw(output_frames)
        output_frames = ReferenceCourt().draw(output_frames)
        output_frames = PlayerStats().draw(output_frames)

        logger.info('Draw frame number at the top left corner of the frame...')
        for i, frame in enumerate(output_frames):
            cv2.putText(frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        logger.info(f'Drew everything in [{(time.monotonic() - start_time_mono):.2f}] seconds') 

        logger.info('Saving output video...')
        start_time_mono = time.monotonic()
        save_video(output_frames, fps, OUTPUT_VIDEO_PATH)
        logger.info(f'Saved video in [{(time.monotonic() - start_time_mono):.2f}] seconds') 

    logger.info('Finish processing')

if __name__ == "__main__":
    main()

