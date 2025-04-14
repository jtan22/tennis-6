from tennis import (
    read_video, 
    save_video,
    get_bounding_box_center_point,
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
    person_positions_all_frames = player_tracker.dectect_person_positions_all_frames(
        input_frames, 
        read_from_stub=True, 
        stub_path='tracker_stubs/player_detections.pkl')

    print('Processing ball tracker...')
    ball_tracker = BallTracker(model_path='models/ball/train-5l6u-10-64/weights/last-378-213.pt')
    ball_positions_all_frames = ball_tracker.dectect_ball_positions_all_frames(
        input_frames, 
        read_from_stub=True, 
        stub_path='tracker_stubs/ball_detections.pkl')

    raw_ball_positions_all_frames = ball_positions_all_frames

    ball_positions_all_frames = ball_tracker.custom_interpolate_ball_positions(ball_positions_all_frames)

    # print('Frame,Raw X,Raw Y,Interpolated X,Interpolated Y,Bounce,Hit')
    for i, raw_ball_positions in enumerate(raw_ball_positions_all_frames):
        raw_x = ''
        raw_y = ''
        if raw_ball_positions.get(1, []) != []:
            raw_ball_position = get_bounding_box_center_point(raw_ball_positions[1])
            raw_x = int(raw_ball_position[0])
            raw_y = int(raw_ball_position[1])
        interpolated_ball_positions = ball_positions_all_frames[i][1]
        interpolated_ball_position = get_bounding_box_center_point(interpolated_ball_positions)
        interpolated_x = int(interpolated_ball_position[0])
        interpolated_y = int(interpolated_ball_position[1])
        # print(f'{i},{raw_x},{raw_y},{interpolated_x},{interpolated_y},,')

    ball_shot_frame_numbers = ball_tracker.get_ball_shot_frame_indexes(ball_positions_all_frames)
    print(f'Ball shot frame numbers: {ball_shot_frame_numbers}')

    ball_tracker.get_ball_hits(ball_positions_all_frames)

    print('Processing court line detector...')
    court_line_detector = CourtLineDetector('models/keypoints/keypoints_model_resnet50_epoch20_cuda.pth')
    court_keypoints = court_line_detector.predict(input_frames[0])
    court_keypoints = court_line_detector.refine_keypoints(input_frames[0], court_keypoints)

    reference_court = ReferenceCourt(input_frames[0].shape[1], input_frames[0].shape[0], 400)
    court_keypoints, homography_matrix = reference_court.get_homographied_keypoints(court_keypoints)
    inverse_homography_matrix = np.linalg.inv(homography_matrix)

    print('Processing player positions...')
    player_positions_all_frames = player_tracker.filter_out_non_players(
        court_keypoints, 
        person_positions_all_frames)

    print('Converting player and ball positions to reference court coordinates...')
    reference_player_coordinates = reference_court.convert_to_reference_court_coordinates(
        player_positions_all_frames, 
        inverse_homography_matrix)
    reference_ball_coordinates = reference_court.convert_to_reference_court_coordinates(
        ball_positions_all_frames, 
        inverse_homography_matrix)
    
    print('Processing player stats...')
    player_stats = PlayerStats(450, reference_court.canvas_width)
    player_stats_data_df = player_stats.collect_stats(
        reference_player_coordinates, 
        reference_ball_coordinates, 
        ball_shot_frame_numbers, 
        fps)
    
    # draw(input_frames, fps, 
    #      player_tracker, player_positions_all_frames, 
    #      ball_tracker, ball_positions_all_frames,
    #      court_line_detector, court_keypoints,
    #      reference_court, reference_player_coordinates, reference_ball_coordinates,
    #      player_stats, player_stats_data_df)

def draw(input_frames, fps,
         player_tracker, player_positions_all_frames, 
         ball_tracker, raw_ball_positions_all_frames, 
         court_line_detector, court_keypoints,
         reference_court, reference_player_coordinates, reference_ball_coordinates,
         player_stats, player_stats_data_df):
    print('Drawing player, ball, keypoints, reference court and player stats on the video...')
    output_frames = player_tracker.draw(input_frames, player_positions_all_frames)
    output_frames = ball_tracker.draw(output_frames, raw_ball_positions_all_frames)
    output_frames = court_line_detector.draw(output_frames, court_keypoints)
    output_frames = reference_court.draw(output_frames)
    output_frames = reference_court.draw_coordinates(output_frames, reference_player_coordinates, 1, 10, color=(0, 0, 255))
    output_frames = reference_court.draw_coordinates(output_frames, reference_player_coordinates, 2, 10, color=(255, 0, 0))
    output_frames = reference_court.draw_coordinates(output_frames, reference_ball_coordinates, 1, 5, color=(0, 255, 0))
    output_frames = player_stats.draw(output_frames, player_stats_data_df)

    print('Draw frame number at the top left corner of the frame...')
    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    print('Saving output video...')
    save_video(output_frames, fps, 'output_videos/output_video.avi')

if __name__ == "__main__":
    main()

