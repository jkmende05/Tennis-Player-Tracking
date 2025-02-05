import cv2
import numpy as np

import sys
sys.path.append('../')

import constants

from utils import convert_pixels_to_meters, convert_meters_to_pixels, get_foot_position, get_closest_keypoint_index, get_box_height, measure_xy_distance, get_center_of_box, get_distance

class MiniCourt():
    """ Class to display mini court and player positions in the top right of the video
    """
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500

        self.buffer = 50
        self.padding = 20

        self.set_canvas_background_position(frame)
        self.set_mini_court_positions()
        self.set_court_points()
        self.set_court_lines()

    def convert_meters(self, meters):
        # Convert meters to pixels using double line width as a reference
        return convert_meters_to_pixels(meters, constants.DOUBLE_LINE_WIDTH, self.court_drawing_width)

    def set_court_points(self):
        drawing_key_points = [0]*28

        # Point 0
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int (self.court_start_y)

        # Point 1
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int (self.court_start_y)

        # Point 2
        drawing_key_points[4], drawing_key_points[5] = int(self.court_start_x), int(self.court_start_y + self.convert_meters(constants.HALF_COURT_LINE_HEIGHT * 2))
        
        # Point 3
        drawing_key_points[6], drawing_key_points[7] = int(drawing_key_points[0] + self.court_drawing_width), int(drawing_key_points[5])

        # Point 4
        drawing_key_points[8], drawing_key_points[9] = int(drawing_key_points[0] + self.convert_meters(constants.DOUBLE_ALLEY_DIFFERENCE)), int(drawing_key_points[1])

        # Point 5
        drawing_key_points[10], drawing_key_points[11] = int(drawing_key_points[4] + self.convert_meters(constants.DOUBLE_ALLEY_DIFFERENCE)), int(drawing_key_points[5])

        # Point 6
        drawing_key_points[12], drawing_key_points[13] = int(drawing_key_points[2] - self.convert_meters(constants.DOUBLE_ALLEY_DIFFERENCE)), int(drawing_key_points[3])

        # Point 7
        drawing_key_points[14], drawing_key_points[15] = int(drawing_key_points[6] - self.convert_meters(constants.DOUBLE_ALLEY_DIFFERENCE)), int(drawing_key_points[7])

        # Point 8
        drawing_key_points[16], drawing_key_points[17] = int(drawing_key_points[8]), int(drawing_key_points[9] + self.convert_meters(constants.NO_MANS_LAND_HEIGHT))

        # Point 9
        drawing_key_points[18], drawing_key_points[19] = int(drawing_key_points[16] + self.convert_meters(constants.SINGLE_LINE_WIDTH)), int(drawing_key_points[17])

        # Point 10
        drawing_key_points[20], drawing_key_points[21] = int(drawing_key_points[10]), int(drawing_key_points[11] - self.convert_meters(constants.NO_MANS_LAND_HEIGHT))

        # Point 11
        drawing_key_points[22], drawing_key_points[23] = int(drawing_key_points[20] + self.convert_meters(constants.SINGLE_LINE_WIDTH)), int(drawing_key_points[21])

        # Point 12
        drawing_key_points[24], drawing_key_points[25] = int((drawing_key_points[16] + drawing_key_points[18]) / 2), int(drawing_key_points[17])

        # Point 13
        drawing_key_points[26], drawing_key_points[27] = int((drawing_key_points[20] + drawing_key_points[22]) / 2), int(drawing_key_points[21])

        self.drawing_key_points = drawing_key_points

    def set_court_lines(self):
        self.lines = [(0, 2), (4, 5), (6, 7), (1, 3), (0, 1), (8, 9), (10, 11), (2, 3)]

    def set_mini_court_positions(self):
        self.court_start_x = self.start_x + self.padding
        self.court_end_x = self.end_x - self.padding

        self.court_start_y = self.start_y + self.padding
        self.court_end_y = self.end_y - self.padding

        self.court_drawing_width = self.court_end_x - self.court_start_x


    def set_canvas_background_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height

        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_background(self, frame):
        # Set background
        shapes = np.zeros_like(frame, np.uint8)

        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)

        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out
    
    def draw_court(self, frame):
        # Draw court lines
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])

            cv2.circle(frame, (x,y), 5, (0, 0, 255), -1)

        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2 + 1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2 + 1]))

            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        net_start = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start, net_end, (255, 0, 0), 2)

        return frame
    
    def draw_mini_court(self, frames):
        output_frames = []
        
        for frame in frames:
            frame = self.draw_background(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)

        return output_frames

    def get_start_point(self):
        return (self.court_start_x, self.court_start_y)
    
    def get_width(self):
        return (self.court_drawing_width)
    
    def get_court_drawing_key_points(self):
        return self.drawing_key_points
    
    def get_mini_court_coordinates(self, object_position, closest_keypoint, closest_keypoint_index, player_height_pixels, player_height_meters):
        x_distance_keypoint_pixels, y_distance_keypoint_pixels = measure_xy_distance(object_position, closest_keypoint)

        x_distance_keypoint_meters = convert_pixels_to_meters(x_distance_keypoint_pixels, player_height_meters, player_height_pixels)
        y_distance_keypoint_meters = convert_pixels_to_meters(y_distance_keypoint_pixels, player_height_meters, player_height_pixels)

        mini_court_x_distance = self.convert_meters(x_distance_keypoint_meters)
        mini_court_y_distance = self.convert_meters(y_distance_keypoint_meters)

        closest_court_keypoint = (self.drawing_key_points[closest_keypoint_index * 2], self.drawing_key_points[closest_keypoint_index * 2  + 1])

        player_position = (closest_court_keypoint[0] + mini_court_x_distance, closest_court_keypoint[1] + mini_court_y_distance)
        return player_position
    
    def convert_boxes_to_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
        player_height = {1 : constants.PLAYER_ONE_HEIGHT, 2 : constants.PLAYER_TWO_HEIGHT}

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, box in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_of_box(ball_box)
            closest_player_id = min(box.keys(), key = lambda x: get_distance(ball_position, get_foot_position(box[x])))

            output_player_boxes_dict = {}
            
            for player_id, bbox in box.items():
                foot_position = get_foot_position(bbox)

                closest_key_points_index = get_closest_keypoint_index(foot_position, original_court_key_points, [0, 2, 12, 13])
                closest_keypoint = (original_court_key_points[closest_key_points_index * 2], original_court_key_points[closest_key_points_index * 2 + 1])

                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)

                box_pixel_height = [get_box_height(player_boxes[i][player_id]) for i in range(frame_index_min, frame_index_max)]
                max_pixel_player_height = max(box_pixel_height)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position, closest_keypoint, closest_key_points_index, max_pixel_player_height, player_height[player_id])

                output_player_boxes_dict[player_id] = mini_court_player_position

                if closest_player_id == player_id:
                    closest_key_points_index = get_closest_keypoint_index(ball_position, original_court_key_points, [0, 2, 12, 13])
                    closest_keypoint = (original_court_key_points[closest_key_points_index * 2], original_court_key_points[closest_key_points_index * 2 + 1])

                    mini_court_player_position = self.get_mini_court_coordinates(foot_position, closest_keypoint, closest_key_points_index, max_pixel_player_height, player_height[player_id])

                    output_ball_boxes.append({1:mini_court_player_position})
            output_player_boxes.append(output_player_boxes_dict)

        return output_player_boxes, output_ball_boxes

    def draw_points(self, frames, positions, color = (0, 255, 0)):
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x, y = position
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x,y), 5, color, -1)

        return frames