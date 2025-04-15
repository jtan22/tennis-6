from tennis import (
    read_video, 
    save_video,
    PlayerTracker,
    BallTracker,
    CourtLineDetector,
    ReferenceCourt,
    PlayerStats)
import cv2
import numpy as np

def main():
    input_video_path = 'input_videos/sample001.mp4'
    input_frames, fps = read_video(input_video_path)

    print('Processing player tracker...')
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_tracker.dectect_person_positions_all_frames(input_frames, read_from_stub=True, stub_path='tracker_stubs/player_detections.pkl')

    print('Processing ball tracker...')
    ball_tracker = BallTracker(model_path='models/ball/train-5l6u-10-64/weights/last-378-213.pt')
    ball_tracker.dectect_ball_positions_all_frames(input_frames, read_from_stub=True, stub_path='tracker_stubs/ball_detections.pkl')
    ball_tracker.remove_all_extra_balls_detected()
    ball_tracker.custom_interpolate_ball_positions()
    ball_tracker.find_ball_shot_frame_numbers()
    ball_tracker.get_ball_hits()

    print('Processing court line detector...')
    court_line_detector = CourtLineDetector('models/keypoints/keypoints_model_resnet50_epoch20_cuda.pth')
    court_line_detector.predict_keypoints(input_frames[0])
    court_line_detector.refine_keypoints(input_frames[0])

    reference_court = ReferenceCourt(input_frames[0].shape[1], input_frames[0].shape[0], 400)
    reference_court.homograph_keypoints(court_line_detector.refined_keypoints)
    court_line_detector.set_homographied_keypoints(reference_court.homographied_keypoints)

    print('Processing player positions...')
    player_tracker.filter_out_non_players(reference_court.homographied_keypoints)

    print('Converting player and ball positions to reference court coordinates...')
    reference_court.convert_player_coordinates(player_tracker.player_positions)
    reference_court.convert_ball_coordinates(ball_tracker.complete_ball_positions)
    
    print('Processing player stats...')
    player_stats = PlayerStats(450, reference_court.canvas_width)
    player_stats.collect_stats(reference_court.player_coordinates, reference_court.ball_coordinates, ball_tracker.ball_shot_frame_numbers, fps)
    
    draw(input_frames, fps, player_tracker, ball_tracker, court_line_detector, reference_court, player_stats)

def draw(input_frames, fps, player_tracker, ball_tracker, court_line_detector, reference_court, player_stats):
    print('Drawing player, ball, keypoints, reference court and player stats on the video...')
    output_frames = player_tracker.draw(input_frames)
    output_frames = ball_tracker.draw(output_frames)
    output_frames = court_line_detector.draw(output_frames)
    output_frames = reference_court.draw(output_frames)
    output_frames = player_stats.draw(output_frames)

    print('Draw frame number at the top left corner of the frame...')
    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    print('Saving output video...')
    save_video(output_frames, fps, 'output_videos/output_video.avi')

if __name__ == "__main__":
    main()

