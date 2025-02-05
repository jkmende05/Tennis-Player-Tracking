from utils import read_video, save_video, get_distance, draw_player_stats, convert_meters_to_pixels, convert_pixels_to_meters
from trackers import PlayerTracker, BallTracker
from court_detection import CourtLineDetector
from mini_court import MiniCourt
import constants

from copy import deepcopy
import pandas as pd
import cv2

def main():
    # Read the input video
    input_videopath = "input_files/input_video_two.mp4"
    input_video_frames = read_video(input_videopath)

    # Detect tennis players and tennis ball
    player_tracker = PlayerTracker(model_path='yolo11x')
    player_detections = player_tracker.detect_frames(input_video_frames, read_from_stub=True, stub_path='tracker_stubs/player_detections.pkl')

    ball_tracker = BallTracker(model_path='models/last.pt')
    ball_detections = ball_tracker.detect_frames(input_video_frames, read_from_stub=True, stub_path='tracker_stubs/ball_detections.pkl')

    ball_detections = ball_tracker.interpolate_ball_position(ball_detections)

    # Detect Court Keypoints
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(model_path=court_model_path)

    court_keypoints = court_line_detector.predict(input_video_frames[0])

    
    player_detections = player_tracker.choose_filter_player(court_keypoints, player_detections)

    # Determine player and ball positions on mini court in top right of video
    mini_court = MiniCourt(input_video_frames[0])   

    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_boxes_to_coordinates(player_detections, ball_detections, court_keypoints)

    # Get player stats, such as shot speed and player speed
    player_stats_data = [{
        'frame_num' : 0,
        'player_1_number_of_shots' : 0,
        'player_1_last_shot_speed' : 0,
        'player_1_total_shot_speed' : 0,
        'player_1_last_player_speed' : 0,
        'player_1_total_player_speed' : 0,

        'player_2_number_of_shots' : 0,
        'player_2_last_shot_speed' : 0,
        'player_2_last_player_speed' : 0,
        'player_2_total_shot_speed' : 0,
        'player_2_total_player_speed' : 0
    }]

    # Detect Ball Shots
    shot_frames = ball_tracker.get_hit_frames(ball_detections)

    for ball_shot_ind in range(len(shot_frames)-1):
        start_frame = shot_frames[ball_shot_ind]
        end_frame = shot_frames[ball_shot_ind + 1]

        ball_shot_time_sec = (end_frame - start_frame) / 25

        pixel_distance_covered = get_distance(ball_mini_court_detections[start_frame][1], ball_mini_court_detections[end_frame][1])

        meter_distance_covered = convert_pixels_to_meters(pixel_distance_covered, constants.DOUBLE_LINE_WIDTH, mini_court.get_width())

        ball_speed = meter_distance_covered / ball_shot_time_sec * 3.6

        player_positions = player_mini_court_detections[start_frame]
        
        player_one = min(player_positions.keys(), key = lambda player_id : get_distance(player_positions[player_id], ball_mini_court_detections[start_frame][1]))

        player_two = 1 if player_one == 2 else 2

        distance_pixels_player_two = get_distance(player_mini_court_detections[start_frame][player_two], player_mini_court_detections[end_frame][player_two])
        distance_meters_player_two = convert_pixels_to_meters(distance_pixels_player_two, constants.DOUBLE_LINE_WIDTH, mini_court.get_width())

        speed_player_two = distance_meters_player_two / ball_shot_time_sec * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame

        current_player_stats[f'player_{player_one}_number_of_shots'] += 1
        current_player_stats[f'player_{player_one}_total_shot_speed'] += ball_speed
        current_player_stats[f'player_{player_one}_last_shot_speed'] = ball_speed

        current_player_stats[f'player_{player_two}_last_player_speed'] = speed_player_two
        current_player_stats[f'player_{player_two}_total_player_speed'] += speed_player_two

        player_stats_data.append(current_player_stats)

    player_stats_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num' : list(range(len(input_video_frames)))})
    player_stats_df = pd.merge(frames_df, player_stats_df, on='frame_num', how='left')
    player_stats_df = player_stats_df.ffill()

    player_stats_df['player_1_average_shot_speed'] = player_stats_df['player_1_total_shot_speed'] / player_stats_df['player_1_number_of_shots']
    player_stats_df['player_1_average_player_speed'] = player_stats_df['player_1_total_player_speed'] / player_stats_df['player_2_number_of_shots']

    player_stats_df['player_2_average_shot_speed'] = player_stats_df['player_2_total_shot_speed'] / player_stats_df['player_2_number_of_shots']
    player_stats_df['player_2_average_player_speed'] = player_stats_df['player_2_total_player_speed'] / player_stats_df['player_1_number_of_shots']

    # Draw Boundary Boxes
    video_frames = player_tracker.draw_boxes(input_video_frames, player_detections)
    video_frames = ball_tracker.draw_boxes(video_frames, ball_detections)

    # Draw Court Keypoints on Video
    video_frames = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)

    video_frames = mini_court.draw_mini_court(video_frames)
    video_frames = mini_court.draw_points(video_frames, player_mini_court_detections)
    video_frames = mini_court.draw_points(video_frames, ball_mini_court_detections, color=(0, 255, 255))

    video_frames = draw_player_stats(video_frames, player_stats_df)

    for i, frame in enumerate(video_frames):
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save output video
    save_video(video_frames, "output_files/output_video.avi")

if __name__  == "__main__":
    main()