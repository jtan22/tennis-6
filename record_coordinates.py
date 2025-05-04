import cv2
import pandas as pd

def main():
    video_path = 'output_videos/output_video.avi'
    csv_path = 'analysis/ball_positions.csv'

    df = pd.read_csv(csv_path)
    coordinates = df['complete_ball_position'].tolist()

    cap = cv2.VideoCapture(video_path)

    current_frame_index = 0

    def update_coordinate(event, x, y, flags, param):
        nonlocal current_frame_index, coordinates, df
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates[current_frame_index] = (x-10, y-10, x+10, y+10)
            df['complete_ball_position'] = coordinates
            df.to_csv(csv_path, index=False)
            print(f"Updated coordinate for frame {current_frame_index + 1}: ({x},{y})")
            show_frame()

    cv2.namedWindow('Video Player')
    cv2.setMouseCallback('Video Player', update_coordinate)

    def show_frame():
        nonlocal current_frame_index
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        ret, frame = cap.read()
        if ret:
            coordinate_text = coordinates[current_frame_index]
            cv2.putText(frame, f"Coordinate: {coordinate_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Video Player', frame)
        else:
            print("Error reading frame.")

    show_frame()

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('a') or key == 81:  # 'a' or left arrow key (81) for previous frame
            current_frame_index = max(0, current_frame_index - 1)
            show_frame()
        elif key == ord('d') or key == 83:  # 'd' or right arrow key (83) for next frame
            current_frame_index = min(len(coordinates) - 1, current_frame_index + 1)
            show_frame()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()