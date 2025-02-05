from ultralytics import YOLO

import cv2
import pickle

import pandas as pd

class BallTracker:
    """ Predicts and determines the position of the ball throughout the video and draws the 
    bounding box within the video
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_position(self, ball_positions):
        # If ball position is unknown, predicts the position based on previous and next known detected ball position
        ball_positions = [x.get(1, []) for x in ball_positions]

        ball_positions_df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        ball_positions_df = ball_positions_df.interpolate()
        ball_positions_df = ball_positions_df.bfill()

        ball_positions = [{1:x} for x in ball_positions_df.to_numpy().tolist()]

        return ball_positions
    
    def get_hit_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        ball_positions_df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        ball_positions_df['mid_y'] = (ball_positions_df['y1'] + ball_positions_df['y2']) / 2
        ball_positions_df['mid_y_rolling_mean'] = ball_positions_df['mid_y'].rolling(window=5, min_periods=1, center=False).mean()

        ball_positions_df['delta_y'] = ball_positions_df['mid_y_rolling_mean'].diff()

        ball_positions_df['ball_hit']=0
        minimum_change_frames_for_hit = 10

        for i in range(1, len(ball_positions_df) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = ball_positions_df['delta_y'].iloc[i] > 0 and ball_positions_df['delta_y'].iloc[i+1] < 0
            positive_position_change = ball_positions_df['delta_y'].iloc[i] < 0 and ball_positions_df['delta_y'].iloc[i+1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i+1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = ball_positions_df['delta_y'].iloc[i] > 0 and ball_positions_df['delta_y'].iloc[change_frame] < 0
                    positive_position_change_following_frame = ball_positions_df['delta_y'].iloc[i] < 0 and ball_positions_df['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                # Only mark the ball hit if there are enough consecutive frames with direction change
                if change_count > minimum_change_frames_for_hit - 1:
                    ball_positions_df['ball_hit'].iloc[i] = 1

        ball_hit_frames = ball_positions_df[ball_positions_df['ball_hit'] == 1].index.tolist()

        return ball_hit_frames 
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf = 0.15)[0]

        ball_dict = {}

        for box in results.boxes:
            result = box.xyxy.tolist()[0]

            ball_dict[1] = result

        return ball_dict
    
    def draw_boxes(self, video_frames, player_detections):
        output_frames = []

        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Boundary Boxes on Players
            for track_id, box in ball_dict.items():
                x1, y1, x2, y2 = box

                cv2.putText(frame, f"Ball ID: {track_id}", (int(box[0]), int(box[1] -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            output_frames.append(frame)

        return output_frames



