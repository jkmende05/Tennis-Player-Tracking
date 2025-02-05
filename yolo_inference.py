from ultralytics import YOLO

model = YOLO('yolov8x.pt')

result = model.track('input_files/input_video.mp4', conf=0.2, save = True)

# print(result)

# print("Boundary Boxes:")
# for box in result[0].boxes:
#     print(box)