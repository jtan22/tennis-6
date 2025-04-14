import pandas as pd
import cv2
from copy import deepcopy
from .constants import STATS_MARGIN_X, STATS_MARGIN_Y, DOUBLES_LINE_WIDTH, SIDE_RUN_WIDTH

class PlayerStats():
    def __init__(self, width, reference_court_canvas_width):
        self.canvas_width = width
        self.canvas_height = int(width / 2)
        self.canvas_x1 = STATS_MARGIN_X
        self.canvas_y1 = STATS_MARGIN_Y
        self.canvas_x2 = self.canvas_x1 + self.canvas_width
        self.canvas_y2 = self.canvas_y1 + self.canvas_height
        self.meter_to_pixel_ratio = (DOUBLES_LINE_WIDTH + SIDE_RUN_WIDTH * 2) / reference_court_canvas_width

    def get_distance_between_coordinates(self, c1, c2):
        """
        Calculates the distance between two players using their coordinates.
        """
        distance = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
        return distance

    def get_player_number_who_hit_ball(self, players_coordinates, ball_coordinates, frame_num):
        distance1 = self.get_distance_between_coordinates(
            players_coordinates[frame_num][1], ball_coordinates[frame_num][1])
        distance2 = self.get_distance_between_coordinates(
            players_coordinates[frame_num][2], ball_coordinates[frame_num][1])
        
        if distance1 < distance2:
            return 1, 2
        else:
            return 2, 1

    def get_player_distance(self, players_coordinates, frame_num, player_number):
        coordinate1 = players_coordinates[frame_num][player_number]
        coordinate2 = players_coordinates[frame_num + 1][player_number]
        distance = self.get_distance_between_coordinates(coordinate1, coordinate2)
        return distance * self.meter_to_pixel_ratio
    
    def get_ball_speed(self, ball_coordinates, frame_num, fps):
        coordinate1 = ball_coordinates[frame_num][1]
        coordinate2 = ball_coordinates[frame_num + 1][1]
        distance = self.get_distance_between_coordinates(coordinate1, coordinate2)
        speed_per_second = distance * self.meter_to_pixel_ratio * fps
        # Convert to km/h
        return speed_per_second * 3.6

    def collect_stats(self, mini_player_coordinates, mini_ball_coordinates, ball_shot_frame_numbers, fps):
        """
        Collects player stats from the mini court coordinates and ball shot frame numbers.
        """
        player_stats_data = [{
            'frame_num':0,
            'player_1_number_of_shots':0,
            'player_1_last_shot_speed':0,
            'player_1_total_shot_speed':0,
            'player_1_last_player_distance':0,
            'player_1_total_player_distance':0,
            'player_2_number_of_shots':0,
            'player_2_last_shot_speed':0,
            'player_2_total_shot_speed':0,
            'player_2_last_player_distance':0,
            'player_2_total_player_distance':0,
        }]

        for shot in range(len(ball_shot_frame_numbers) - 1):
            hitting_player, receiving_player = self.get_player_number_who_hit_ball(
                mini_player_coordinates, mini_ball_coordinates, ball_shot_frame_numbers[shot])
            for frame_num in range(ball_shot_frame_numbers[shot], ball_shot_frame_numbers[shot + 1] - 1):
                player_stats = deepcopy(player_stats_data[-1])
                player_stats['frame_num'] = frame_num
                if frame_num == ball_shot_frame_numbers[shot]:
                    player_stats[f'player_{hitting_player}_number_of_shots'] += 1
                
                last_shot_speed = self.get_ball_speed(mini_ball_coordinates, frame_num, fps)
                player_stats[f'player_{hitting_player}_last_shot_speed'] = last_shot_speed
                player_stats[f'player_{hitting_player}_total_shot_speed'] += last_shot_speed

                player_distance = self.get_player_distance(mini_player_coordinates, frame_num, hitting_player)                
                player_stats[f'player_{hitting_player}_last_player_distance'] = player_distance
                player_stats[f'player_{hitting_player}_total_player_distance'] += player_distance

                player_distance = self.get_player_distance(mini_player_coordinates, frame_num, receiving_player)
                player_stats[f'player_{receiving_player}_last_player_distance'] = player_distance
                player_stats[f'player_{receiving_player}_total_player_distance'] += player_distance
                player_stats_data.append(player_stats)

        player_stats_data_df = pd.DataFrame(player_stats_data)
        frames_df = pd.DataFrame({'frame_num': list(range(len(mini_player_coordinates)))})
        player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
        player_stats_data_df = player_stats_data_df.ffill()

        return player_stats_data_df
    
    def draw(self, input_frames, player_stats_data_df):
        output_frames = []
        for index, row in player_stats_data_df.iterrows():
            frame = input_frames[index]
            overlay = frame.copy()
            cv2.rectangle(overlay, (self.canvas_x1, self.canvas_y1), (self.canvas_x2, self.canvas_y2), (0, 0, 0), -1)
            alpha = 0.5 
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            text = "Player 1     Player 2"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 200, self.canvas_y1 + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            text = "Last Shot Speed"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 10, self.canvas_y1 + 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            text = f"{row['player_1_last_shot_speed']:.1f} km/h      {row['player_2_last_shot_speed']:.1f} km/h"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 200, self.canvas_y1 + 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            text = "Total Shot Speed"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 10, self.canvas_y1 + 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            text = f"{row['player_1_total_shot_speed']:.1f} km/h      {row['player_2_total_shot_speed']:.1f} km/h"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 200, self.canvas_y1 + 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)            
            
            text = "Last Player Distance"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 10, self.canvas_y1 + 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            text = f"{row['player_1_last_player_distance']:.1f} m          {row['player_2_last_player_distance']:.1f} m"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 200, self.canvas_y1 + 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            text = "Total Player Distance"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 10, self.canvas_y1 + 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            text = f"{row['player_1_total_player_distance']:.1f} m          {row['player_2_total_player_distance']:.1f} m"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 200, self.canvas_y1 + 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            output_frames.append(frame)
        
        return output_frames

