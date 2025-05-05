import math
from typing import List, Tuple

# Constances for tennis court dimensions and physics
SINGLES_LINE_WIDTH  = 8.22
DOUBLES_LINE_WIDTH  = 10.97
HALF_COURT_DEPTH    = 11.88
SERVICE_LINE_DEPTH  = 6.4
DOUBLES_ALLEY_WIDTH = 1.37
NO_MANS_LAND_DEPTH  = 5.48
RUN_BACK_DEPTH      = 4.1
SIDE_RUN_WIDTH      = 2.3
CENTER_LINE_DEPTH   = 0.1
NET_CENTRE_HEIGHT   = 0.91
COURT_KEYPOINTS     = 14

# Constants for reference court mini window
REFERENCE_COURT_MARGIN_X = 50
REFERENCE_COURT_MARGIN_Y = 50
REFERENCE_COURT_CANVAS_WIDTH = 400
REFERENCE_COURT_PIXEL_TO_METER_RATIO = REFERENCE_COURT_CANVAS_WIDTH / (DOUBLES_LINE_WIDTH + 2 * SIDE_RUN_WIDTH)
REFERENCE_COURT_CANVAS_DEPTH = round(((HALF_COURT_DEPTH + RUN_BACK_DEPTH) * 2) * REFERENCE_COURT_PIXEL_TO_METER_RATIO)
REFERENCE_COURT_X1 = round(SIDE_RUN_WIDTH * REFERENCE_COURT_PIXEL_TO_METER_RATIO)
REFERENCE_COURT_X2 = REFERENCE_COURT_CANVAS_WIDTH - REFERENCE_COURT_X1
REFERENCE_COURT_Y1 = round(RUN_BACK_DEPTH * REFERENCE_COURT_PIXEL_TO_METER_RATIO)
REFERENCE_COURT_Y2 = REFERENCE_COURT_CANVAS_DEPTH - REFERENCE_COURT_Y1
REFERENCE_COURT_DOUBLE_ALLEY_WIDTH = round(DOUBLES_ALLEY_WIDTH * REFERENCE_COURT_PIXEL_TO_METER_RATIO)
REFERENCE_COURT_NO_MANS_LAND_DEPTH = round(NO_MANS_LAND_DEPTH * REFERENCE_COURT_PIXEL_TO_METER_RATIO)
# 0-4---------6-1
# | |         | |
# | |         | |
# | 8----C----9 |
# | |    |    | |
# | |    |    | |
# +-+----+----+-+
# | |    |    | |
# | |    |    | |
# | A----D----B |
# | |         | |
# | |         | |
# 2-5---------7-3
REFERENCE_KEYPOINTS: List[Tuple[int, int]] = [
    (REFERENCE_COURT_X1, REFERENCE_COURT_Y1),  # 0
    (REFERENCE_COURT_X2, REFERENCE_COURT_Y1),  # 1
    (REFERENCE_COURT_X1, REFERENCE_COURT_Y2),  # 2
    (REFERENCE_COURT_X2, REFERENCE_COURT_Y2),  # 3
    (REFERENCE_COURT_X1 + REFERENCE_COURT_DOUBLE_ALLEY_WIDTH, REFERENCE_COURT_Y1),  # 4
    (REFERENCE_COURT_X1 + REFERENCE_COURT_DOUBLE_ALLEY_WIDTH, REFERENCE_COURT_Y2),  # 5
    (REFERENCE_COURT_X2 - REFERENCE_COURT_DOUBLE_ALLEY_WIDTH, REFERENCE_COURT_Y1),  # 6
    (REFERENCE_COURT_X2 - REFERENCE_COURT_DOUBLE_ALLEY_WIDTH, REFERENCE_COURT_Y2),  # 7
    (REFERENCE_COURT_X1 + REFERENCE_COURT_DOUBLE_ALLEY_WIDTH, REFERENCE_COURT_Y1 + REFERENCE_COURT_NO_MANS_LAND_DEPTH),  # 8
    (REFERENCE_COURT_X2 - REFERENCE_COURT_DOUBLE_ALLEY_WIDTH, REFERENCE_COURT_Y1 + REFERENCE_COURT_NO_MANS_LAND_DEPTH),  # 9
    (REFERENCE_COURT_X1 + REFERENCE_COURT_DOUBLE_ALLEY_WIDTH, REFERENCE_COURT_Y2 - REFERENCE_COURT_NO_MANS_LAND_DEPTH),  # 10
    (REFERENCE_COURT_X2 - REFERENCE_COURT_DOUBLE_ALLEY_WIDTH, REFERENCE_COURT_Y2 - REFERENCE_COURT_NO_MANS_LAND_DEPTH),  # 11
    (round((REFERENCE_COURT_X1 + REFERENCE_COURT_X2) / 2), REFERENCE_COURT_Y1 + REFERENCE_COURT_NO_MANS_LAND_DEPTH),  # 12
    (round((REFERENCE_COURT_X1 + REFERENCE_COURT_X2) / 2), REFERENCE_COURT_Y2 - REFERENCE_COURT_NO_MANS_LAND_DEPTH)   # 13
]
# 0-4---------6-1
# | |         | |
# | |         | |
# | 8----C----9 |
# | |    |    | |
# | |    |    | |
# +-+----+----+-+
# | |    |    | |
# | |    |    | |
# | A----D----B |
# | |         | |
# | |         | |
# 2-5---------7-3
REFERENCE_COURT_LINES = [
    (0, 2),   # Left sideline
    (4, 5),   # Left singles sideline
    (6, 7),   # Right singles sideline
    (1, 3),   # Right sideline
    (0, 1),   # Top baseline
    (8, 9),   # Top service line
    (12, 13), # Center service line
    (10, 11), # Bottom service line
    (2, 3)    # Bottom baseline
]
# 0-4---------6-1
# | |         | |
# | |         | |
# | 8----C----9 |
# | |    |    | |
# | |    |    | |
# +-+----+----+-+
# | |    |    | |
# | |    |    | |
# | A----D----B |
# | |         | |
# | |         | |
# 2-5---------7-3
COURT_HOMOGRAPHY_CONFIGURATIONS = [
    [0, 1, 2, 3],    # doubles court
    [0, 6, 2, 7],    # right doubles alley plus singles court
    [4, 1, 5, 3],    # left doubles alley plus singles court
    [4, 6, 5, 7],    # singles court
    [0, 4, 2, 5],    # right doubles alley
    [6, 1, 7, 3],    # left doubles alley
    [4, 6, 10, 11],  # top no mans land plus mini court
    [10, 11, 5, 7]   # bottom no mans land
]

SIDE_VIEW_COURT_CANVAS_HEIGHT = 100

# Constants for player stats mini window
PLAYER_STATS_WIDTH      = 500
PLAYER_STATS_HEIGHT     = 400
PLAYER_STATS_MARGIN_X   = 50
PLAYER_STATS_MARGIN_Y   = 50

# Constants for ball physics
BALL_RADIUS                     = 0.033
BALL_DRAG_COEFFICIENT           = 0.6
AIR_DENSITY                     = 1.225
BALL_CROSS_SECTION              = math.pi * BALL_RADIUS ** 2
BALL_MASS                       = 0.057
BALL_DRAG_FACTOR                = AIR_DENSITY * BALL_DRAG_COEFFICIENT * BALL_CROSS_SECTION
GRAVITY                         = 9.8
BALL_TERMINAL_VELOCITY          = ((2 * BALL_MASS * GRAVITY) / (BALL_DRAG_COEFFICIENT * AIR_DENSITY * BALL_CROSS_SECTION)) ** 0.5
BALL_TERMINAL_VELOCITY_SQUARED  = BALL_TERMINAL_VELOCITY ** 2
DEFAULT_VERTICAL_VELOCITY       = 3.0

RESNET50_PRE_TRAINED_MODEL_IMAGE_SIZE = 224

# Constants for ball states
BALL_UNKNOWN     = -1
BALL_FAR_HIT     = 0
BALL_NEAR_BOUNCE = 1
BALL_NEAR_HIT    = 2
BALL_FAR_BOUNCE  = 3
BALL_IN_FLIGHT   = 4
BALL_DEAD        = 5

# Constants for player states
PLAYER_UNKNOWN = -1
PLAYER_RUNNING = 0
PLAYER_HITTING = 1

NEAR_PLAYER_NAME = 'DJOKOVIC'
FAR_PLAYER_NAME  = 'MENSIK'
