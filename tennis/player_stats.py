import pandas as pd
import cv2
from copy import deepcopy
from typing import List
import numpy as np

from tennis.reference_ball import ReferenceBall
from tennis.reference_frame import ReferenceFrame
from .constants import (
    FAR_PLAYER_NAME, 
    NEAR_PLAYER_NAME, 
    PLAYER_HITTING, 
    PLAYER_STATS_MARGIN_X, 
    PLAYER_STATS_MARGIN_Y, 
    REFERENCE_COURT_PIXEL_TO_METER_RATIO, 
    PLAYER_STATS_WIDTH,
    PLAYER_STATS_HEIGHT,
    REFERENCE_COURT_CANVAS_DEPTH,
)
from tennis.utils import get_distance_between_points

class PlayerStats():

    def __init__(self):
        self.player_stats_path = 'analysis/player_stats.csv'

        self.canvas_x1 = PLAYER_STATS_MARGIN_X
        self.canvas_y1 = PLAYER_STATS_MARGIN_Y
        self.canvas_x2 = self.canvas_x1 + PLAYER_STATS_WIDTH
        self.canvas_y2 = self.canvas_y1 + PLAYER_STATS_HEIGHT

    def collect_stats(self, reference_frames: List[ReferenceFrame]) -> None:
        """
        Collects player stats from the mini court coordinates and ball shot frame numbers.
        """
        player_stats_data = [{
            'frame_num':                            0,
            'near_player_serve_speed':              0.0,
            'near_player_number_of_shots':          0,
            'near_player_last_shot_speed':          0.0,
            'near_player_total_shot_speed':         0.0,
            'near_player_number_of_net_clearances': 0,
            'near_player_last_net_clearance':       0.0,
            'near_player_total_net_clearance':      0.0,
            'near_player_last_distance':            0.0,
            'near_player_total_distance':           0.0,
            'far_player_serve_speed':               0.0,
            'far_player_number_of_shots':           0,
            'far_player_last_shot_speed':           0.0,
            'far_player_total_shot_speed':          0.0,
            'far_player_number_of_net_clearances':  0,
            'far_player_last_net_clearance':        0.0,
            'far_player_total_net_clearance':       0.0,
            'far_player_last_distance':             0.0,
            'far_player_total_distance':            0.0,
        }]

        served = False
        near_hit = False
        far_hit = False
        last_ball = reference_frames[0].ball
        for i in range(1, len(reference_frames)):
            player_stats = deepcopy(player_stats_data[-1])
            player_stats['frame_num'] = i

            current_ball = reference_frames[i].ball
            velocity_km_per_hour = round(current_ball.velocity * 3.6, 2)
            if reference_frames[i].near_player.action == PLAYER_HITTING:
                if served:
                    player_stats['near_player_number_of_shots'] += 1
                    player_stats['near_player_last_shot_speed'] = velocity_km_per_hour
                    player_stats['near_player_total_shot_speed'] = round(player_stats['near_player_total_shot_speed'] + velocity_km_per_hour, 2)
                    near_hit = True
                    far_hit = False
                else:
                    player_stats['near_player_serve_speed'] = velocity_km_per_hour
                    served = True
            if reference_frames[i].far_player.action == PLAYER_HITTING:
                if served:
                    player_stats['far_player_number_of_shots'] += 1
                    player_stats['far_player_last_shot_speed'] = velocity_km_per_hour
                    player_stats['far_player_total_shot_speed'] = round(player_stats['far_player_total_shot_speed'] + velocity_km_per_hour, 2)
                    near_hit = False
                    far_hit = True
                else:
                    player_stats['far_player_serve_speed'] = velocity_km_per_hour
                    served = True

            near_player_distance_pixels = get_distance_between_points(reference_frames[i].near_player.reference_coordinate, reference_frames[i - 1].near_player.reference_coordinate)
            player_stats['near_player_last_distance'] = round(near_player_distance_pixels / REFERENCE_COURT_PIXEL_TO_METER_RATIO, 2)
            player_stats['near_player_total_distance'] = round(player_stats['near_player_total_distance'] + player_stats['near_player_last_distance'], 2)

            far_player_distance_pixels = get_distance_between_points(reference_frames[i].far_player.reference_coordinate, reference_frames[i - 1].far_player.reference_coordinate)
            player_stats['far_player_last_distance'] = round(far_player_distance_pixels / REFERENCE_COURT_PIXEL_TO_METER_RATIO, 2)
            player_stats['far_player_total_distance'] = round(player_stats['far_player_total_distance'] + player_stats['far_player_last_distance'], 2)

            net_clearance = self._get_net_clearance(last_ball, current_ball)
            if net_clearance > 0:
                if near_hit:
                    player_stats['near_player_number_of_net_clearances'] += 1
                    player_stats['near_player_last_net_clearance'] = net_clearance
                    player_stats['near_player_total_net_clearance'] = round(player_stats['near_player_total_net_clearance'] + net_clearance, 2)
                elif far_hit:
                    player_stats['far_player_number_of_net_clearances'] += 1
                    player_stats['far_player_last_net_clearance'] = net_clearance
                    player_stats['far_player_total_net_clearance'] = round(player_stats['far_player_total_net_clearance'] + net_clearance, 2)
            last_ball = current_ball

            player_stats_data.append(player_stats)

        df = pd.DataFrame(player_stats_data)
        df.to_csv(self.player_stats_path, index=False)
    
    def _get_net_clearance(self, last_ball: ReferenceBall, current_ball: ReferenceBall) -> float:
        net_y = REFERENCE_COURT_CANVAS_DEPTH // 2
        
        current_y = current_ball.reference_coordinate[1]
        if current_y == net_y:
            return current_ball.height_meters
        
        last_y = last_ball.reference_coordinate[1]
        if (last_y < net_y and net_y < current_y) or (last_y > net_y and net_y > current_y):
            y_ratio = (current_y - net_y) / (current_y - last_y)
            ball_height_diff = current_ball.height_meters - last_ball.height_meters
            return round(current_ball.height_meters - ball_height_diff * y_ratio, 2)

        return 0.0

    def draw(self, input_frames: List[np.ndarray]):
        df = pd.read_csv(self.player_stats_path)
        player_stats_data = df.to_dict(orient='records')
        output_frames = []
        for index, per_frame_data in enumerate(player_stats_data):
            frame = input_frames[index]
            overlay = frame.copy()
            cv2.rectangle(overlay, (self.canvas_x1, self.canvas_y1), (self.canvas_x2, self.canvas_y2), (0, 0, 0), -1)
            alpha = 0.5 
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            frame = cv2.putText(frame, f'{NEAR_PLAYER_NAME}', 
                                (self.canvas_x1 + 200, self.canvas_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, f'{FAR_PLAYER_NAME}', 
                                (self.canvas_x1 + 350, self.canvas_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            self.draw_stat(
                frame, 
                'Serve Speed (km/h)', 
                f'{per_frame_data['near_player_serve_speed']}', 
                f'{per_frame_data['far_player_serve_speed']}', 
                80)

            self.draw_stat(
                frame, 
                'Last Shot Speed (km/h)', 
                f'{per_frame_data['near_player_last_shot_speed']}', 
                f'{per_frame_data['far_player_last_shot_speed']}', 
                120)

            near_player_average_shot_speed = round(per_frame_data['near_player_total_shot_speed'] / max(1, per_frame_data['near_player_number_of_shots']), 2)
            far_player_average_shot_speed = round(per_frame_data['far_player_total_shot_speed'] / max(1, per_frame_data['far_player_number_of_shots']), 2)
            self.draw_stat(
                frame, 
                'Average Shot Speed (km/h)', 
                f'{near_player_average_shot_speed}', 
                f'{far_player_average_shot_speed}', 
                160)

            self.draw_stat(
                frame, 
                'Last Net Clearance (m)', 
                f'{per_frame_data['near_player_last_net_clearance']}', 
                f'{per_frame_data['far_player_last_net_clearance']}', 
                200)

            near_player_average_net_clearance = round(per_frame_data['near_player_total_net_clearance'] / max(1, per_frame_data['near_player_number_of_net_clearances']), 2)
            far_player_average_net_clearance = round(per_frame_data['far_player_total_net_clearance'] / max(1, per_frame_data['far_player_number_of_net_clearances']), 2)
            self.draw_stat(
                frame, 
                'Average Net Clearance (m)', 
                f'{near_player_average_net_clearance}', 
                f'{far_player_average_net_clearance}', 
                240)

            self.draw_stat(
                frame, 
                'Last Player Distance (m)', 
                f'{per_frame_data['near_player_last_distance']}', 
                f'{per_frame_data['far_player_last_distance']}', 
                280)
            
            self.draw_stat(
                frame, 
                'Total Player Distance (m)', 
                f'{per_frame_data['near_player_total_distance']}', 
                f'{per_frame_data['far_player_total_distance']}', 
                320)

            output_frames.append(frame)
        
        return output_frames

    def draw_stat(self, frame: np.ndarray, description: str, near_stat: str, far_stat: str, y: int) -> None:
        frame = cv2.putText(frame, description, (self.canvas_x1 + 10,  self.canvas_y1 + y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        frame = cv2.putText(frame, near_stat,   (self.canvas_x1 + 200, self.canvas_y1 + y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, far_stat,    (self.canvas_x1 + 350, self.canvas_y1 + y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)



