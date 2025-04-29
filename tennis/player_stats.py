import pandas as pd
import cv2
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Any

from sympy import per

from tennis.reference_frame import ReferenceFrame
from .constants import PLAYER_HIT, STATS_MARGIN_X, STATS_MARGIN_Y, DOUBLES_LINE_WIDTH, SIDE_RUN_WIDTH
from tennis.utils import get_distance_between_points

class PlayerStats():
    def __init__(self, width, reference_court_canvas_width):
        self.canvas_width = width
        self.canvas_height = int(width / 2)
        self.canvas_x1 = STATS_MARGIN_X
        self.canvas_y1 = STATS_MARGIN_Y
        self.canvas_x2 = self.canvas_x1 + self.canvas_width
        self.canvas_y2 = self.canvas_y1 + self.canvas_height
        self.meter_to_pixel_ratio = (DOUBLES_LINE_WIDTH + SIDE_RUN_WIDTH * 2) / reference_court_canvas_width

    def collect_stats(self, reference_frames: List[ReferenceFrame]):
        """
        Collects player stats from the mini court coordinates and ball shot frame numbers.
        """
        self.player_stats_data = [{
            'frame_num':0,
            'player_1_last_player_distance':0.0,
            'player_1_total_player_distance':0.0,
            'player_2_last_player_distance':0.0,
            'player_2_total_player_distance':0.0,
            'player_1_number_of_shots':0,
            'player_1_last_shot_speed':0.0,
            'player_1_total_shot_speed':0.0,
            'player_2_number_of_shots':0,
            'player_2_last_shot_speed':0.0,
            'player_2_total_shot_speed':0.0,
        }]

        for i in range(1, len(reference_frames)):
            player_stats = deepcopy(self.player_stats_data[-1])
            player_stats['frame_num'] = i
            player_1_distance_pixels = get_distance_between_points(
                reference_frames[i].player_1.reference_coordinate, 
                reference_frames[i - 1].player_1.reference_coordinate)
            player_1_distance_meters = round(player_1_distance_pixels * self.meter_to_pixel_ratio, 2)
            player_2_distance_pixels = get_distance_between_points(
                reference_frames[i].player_2.reference_coordinate, 
                reference_frames[i - 1].player_2.reference_coordinate)
            player_2_distance_meters = round(player_2_distance_pixels * self.meter_to_pixel_ratio, 2)
            player_stats[f'player_1_last_player_distance'] = player_1_distance_meters
            player_1_total_distance = round(player_stats[f'player_1_total_player_distance'] + player_1_distance_meters, 2)
            player_stats[f'player_1_total_player_distance'] = player_1_total_distance
            player_stats[f'player_2_last_player_distance'] = player_2_distance_meters
            player_2_total_distance = round(player_stats[f'player_2_total_player_distance'] + player_2_distance_meters, 2)
            player_stats[f'player_2_total_player_distance'] = player_2_total_distance
            if reference_frames[i].player_1.action == PLAYER_HIT:
                velocity_meters_per_second = reference_frames[i].ball.velocity
                if velocity_meters_per_second > 0:
                    player_stats[f'player_1_number_of_shots'] += 1
                    velocity_km_per_hour = round(velocity_meters_per_second * 3.6, 2)
                    player_stats[f'player_1_last_shot_speed'] = velocity_km_per_hour
                    player_stats[f'player_1_total_shot_speed'] += velocity_km_per_hour
            if reference_frames[i].player_2.action == PLAYER_HIT:
                velocity_meters_per_second = reference_frames[i].ball.velocity
                if velocity_meters_per_second > 0:
                    player_stats[f'player_2_number_of_shots'] += 1
                    velocity_km_per_hour = round(velocity_meters_per_second * 3.6, 2)
                    player_stats[f'player_2_last_shot_speed'] = velocity_km_per_hour
                    player_stats[f'player_2_total_shot_speed'] += velocity_km_per_hour
            self.player_stats_data.append(player_stats)

        df = pd.DataFrame(self.player_stats_data)
        df.to_csv('player_stats.csv', index=False)
        print("Player stats collected and saved to player_stats.csv")
    
    def draw(self, input_frames):
        output_frames = []
        for index, per_frame_data in enumerate(self.player_stats_data):
            frame = input_frames[index]
            overlay = frame.copy()
            cv2.rectangle(overlay, (self.canvas_x1, self.canvas_y1), (self.canvas_x2, self.canvas_y2), (0, 0, 0), -1)
            alpha = 0.5 
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            text = "Player 1     Player 2"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 200, self.canvas_y1 + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            text = "Last Shot Speed"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 10, self.canvas_y1 + 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            text = f"{per_frame_data['player_1_last_shot_speed']} km/h      {per_frame_data['player_2_last_shot_speed']} km/h"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 200, self.canvas_y1 + 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            text = "Average Shot Speed"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 10, self.canvas_y1 + 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            player_1_average_shot_speed = round(per_frame_data['player_1_total_shot_speed'] / max(1, per_frame_data['player_1_number_of_shots']), 2)
            player_2_average_shot_speed = round(per_frame_data['player_2_total_shot_speed'] / max(1, per_frame_data['player_2_number_of_shots']), 2)
            text = f"{player_1_average_shot_speed} km/h      {player_2_average_shot_speed} km/h"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 200, self.canvas_y1 + 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)            
            
            text = "Last Player Distance"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 10, self.canvas_y1 + 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            text = f"{per_frame_data['player_1_last_player_distance']} m          {per_frame_data['player_2_last_player_distance']} m"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 200, self.canvas_y1 + 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            text = "Total Player Distance"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 10, self.canvas_y1 + 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            text = f"{per_frame_data['player_1_total_player_distance']} m          {per_frame_data['player_2_total_player_distance']} m"
            frame = cv2.putText(frame, text, (self.canvas_x1 + 200, self.canvas_y1 + 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            output_frames.append(frame)
        
        return output_frames

