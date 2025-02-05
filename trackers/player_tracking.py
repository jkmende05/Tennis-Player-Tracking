from ultralytics import YOLO

import cv2
import pickle

import sys
sys.path.append('../')

from utils import get_center, get_distance

class PlayerTracker:
    """ Class to only display the two players with IDs 1 and 2, detect their position within the video,
    and draw the bounding boxes on the video
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_filter_player(self, court_keypoints, player_detections):
        player_detections_frame_one = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detections_frame_one)

        filtered_players = []
        for player_dict in player_detections:
            filtered_players_dict = {track_id: box for track_id, box in player_dict.items() if track_id in chosen_players}
            filtered_players.append(filtered_players_dict)

        return filtered_players

    def choose_players(self, court_keypoints, player_detections_frame_one):
        # Force selection of player IDs 1 and 2
        chosen_players = [1, 2]

        # # If you want to still calculate distances to make sure they are close to the court:
        # distances = []
        # for track_id, box in player_detections_frame_one.items():
        #     player_center = get_center(box)
            
        #     min_distance = float('inf')

        #     for i in range(0, len(court_keypoints), 2):
        #         court_keypoint = (court_keypoints[i], court_keypoints[i+1])
        #         distance = get_distance(player_center, court_keypoint)

        #         if distance < min_distance:
        #             min_distance = distance

        #     distances.append((track_id, min_distance))

        # # Sort players based on their proximity to the court (optional)
        # distances.sort(key = lambda x: x[1])

        # # You can still choose the closest players based on distance if needed
        # if len(distances) >= 2:
        #     chosen_players = [distances[0][0], distances[1][0]]

        return chosen_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}

        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]

            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]

            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict
    
    def draw_boxes(self, video_frames, player_detections):
        output_frames = []

        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Boundary Boxes on Players
            for track_id, box in player_dict.items():
                x1, y1, x2, y2 = box

                cv2.putText(frame, f"Player ID: {track_id}", (int(box[0]), int(box[1] -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            output_frames.append(frame)

        return output_frames



