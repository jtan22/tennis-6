from tennis.ball_analyser import BallAnalyser
from tennis.ball_tracker import BallTracker
from tennis.court_line_detector import CourtLineDetector
from tennis.player_tracker import PlayerTracker
from tennis.reference_court import ReferenceCourt
from tennis.player_stats import PlayerStats
from tennis.utils import read_video, save_video
import cv2

def main():

    INPUT_VIDEO_PATH                = 'input_videos/sample001.mp4'
    OUTPUT_VIDEO_PATH               = 'output_videos/output_video.avi'
    PLAYER_TRACKER_MODEL_PATH       = 'yolov8x'
    BALL_TRACKER_MODEL_PATH         = 'models/ball/train-5l6u-10-64/weights/last-378-213.pt'
    COURT_LINE_DETECTOR_MODEL_PATH  = 'models/keypoints/keypoints_model_resnet50_epoch20_cuda.pth'

    input_frames, fps = read_video(INPUT_VIDEO_PATH)
    fps = round(fps)

    DETECT_PERSON_POSITIONS         = False
    DETECT_BALL_POSITIONS           = False
    SYNTHESIZE_BALL_POSITIONS       = False
    FIND_BALL_HITS_AND_BOUNCES      = False
    DETECT_KEYPOINTS                = False
    FIND_PLAYER_POSITIONS           = False
    COMPUTE_REFERENCE_COORDINATES   = True
    COLLECT_PLAYER_STATS            = True
    DRAW_FRAMES                     = False

    if DETECT_PERSON_POSITIONS:
        print('Detecting person positions...')
        PlayerTracker().dectect_person_positions(input_frames, PLAYER_TRACKER_MODEL_PATH)

    if DETECT_BALL_POSITIONS:
        print('Detecting ball positions...')
        BallTracker().detect_ball_positions(input_frames, BALL_TRACKER_MODEL_PATH)

    if SYNTHESIZE_BALL_POSITIONS:
        print('Synthesizing ball positions...')
        BallTracker().synthesize_ball_positions()

    if FIND_BALL_HITS_AND_BOUNCES:
        print('Finding ball hits and bounces...')
        BallAnalyser().find_ball_hits_and_bounces(fps, BallTracker().load_ball_positions_df())

    if DETECT_KEYPOINTS:
        print('Detecting keypoints...')
        CourtLineDetector().detect_keypoints(input_frames[0], COURT_LINE_DETECTOR_MODEL_PATH)

    if FIND_PLAYER_POSITIONS:
        print('Finding player positions...')
        PlayerTracker().find_player_positions(CourtLineDetector().load_court_keypoints())

    if COMPUTE_REFERENCE_COORDINATES:
        print('Computing reference coordinates...')
        ReferenceCourt().compute_reference_coordinates(
            CourtLineDetector().load_court_keypoints(),
            *PlayerTracker().load_player_positions(), 
            BallTracker().load_complete_ball_positions(), 
            BallAnalyser().load_ball_hits_and_bounces(),
            fps)
    
    if COLLECT_PLAYER_STATS:
        print('Collecting player stats...')
        PlayerStats().collect_stats(ReferenceCourt().load_reference_frames())
    
    if DRAW_FRAMES:
        print('Drawing player, ball, keypoints, reference court and player stats on the video...')
        output_frames = PlayerTracker().draw(input_frames)
        output_frames = BallTracker().draw(output_frames)
        output_frames = CourtLineDetector().draw(output_frames)
        output_frames = ReferenceCourt().draw(output_frames)
        output_frames = PlayerStats().draw(output_frames)

        print('Draw frame number at the top left corner of the frame...')
        for i, frame in enumerate(output_frames):
            cv2.putText(frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        print('Saving output video...')
        save_video(output_frames, fps, OUTPUT_VIDEO_PATH)

if __name__ == "__main__":
    main()

