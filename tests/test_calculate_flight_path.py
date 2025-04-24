import unittest
from unittest.mock import patch
from tennis.reference_court import ReferenceCourt

class TestCalculateFlightPath(unittest.TestCase):
    def setUp(self):
        # Initialize a ReferenceCourt object with mock dimensions
        self.reference_court = ReferenceCourt(frame_width=1920, frame_height=1080, canvas_width=400)
        self.reference_court.pixel_to_meter_ratio = 0.1  # Mock ratio for testing

    @patch('tennis.reference_court.get_distance_between_points', return_value=10)
    @patch('tennis.reference_court.get_initial_horizontal_velocity', return_value=5)
    @patch('tennis.reference_court.get_initial_vertical_velocity', return_value=2)
    @patch('tennis.reference_court.get_vertical_distances_and_velocities', return_value=([0, 1, 2], [0, 1, 2]))
    @patch('tennis.reference_court.get_hypotenuse', return_value=5.39)
    def test_calculate_flight_path_with_height(self, mock_hypotenuse, mock_vertical_distances, mock_vertical_velocity, mock_horizontal_velocity, mock_distance):
        start_point = (100, 200)
        end_point = (200, 300)
        frame_count = 10
        fps = 30
        ball_hit_height = 1.5

        result = self.reference_court._calculate_flight_path(start_point, end_point, frame_count, fps, ball_hit_height)

        # Check the first and last points
        self.assertEqual(result[0], (100, 200, 1.5, 5.39))  # Start point with height and velocity
        self.assertEqual(result[-1][:2], (200, 300))  # End point coordinates
        self.assertEqual(len(result), frame_count)  # Ensure the correct number of frames

    @patch('tennis.reference_court.get_distance_between_points', return_value=10)
    @patch('tennis.reference_court.get_initial_horizontal_velocity', return_value=5)
    def test_calculate_flight_path_without_height(self, mock_horizontal_velocity, mock_distance):
        start_point = (100, 200)
        end_point = (200, 300)
        frame_count = 10
        fps = 30
        ball_hit_height = 0  # No height

        result = self.reference_court._calculate_flight_path(start_point, end_point, frame_count, fps, ball_hit_height)

        # Check the first and last points
        self.assertEqual(result[0], (100, 200))  # Start point without height
        self.assertEqual(result[-1], (200, 300))  # End point coordinates
        self.assertEqual(len(result), frame_count)  # Ensure the correct number of frames

    def test_calculate_flight_path_zero_frames(self):
        start_point = (100, 200)
        end_point = (200, 300)
        frame_count = 0
        fps = 30
        ball_hit_height = 1.5

        result = self.reference_court._calculate_flight_path(start_point, end_point, frame_count, fps, ball_hit_height)

        # Ensure the result is empty
        self.assertEqual(result, [])

    def test_calculate_flight_path_zero_distance(self):
        start_point = (100, 200)
        end_point = (100, 200)  # Same as start point
        frame_count = 10
        fps = 30
        ball_hit_height = 1.5

        result = self.reference_court._calculate_flight_path(start_point, end_point, frame_count, fps, ball_hit_height)

        # Ensure all points are the same
        for point in result:
            self.assertEqual(point[:2], (100, 200))  # Coordinates should not change

if __name__ == '__main__':
    unittest.main()