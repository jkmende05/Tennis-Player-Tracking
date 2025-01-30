from ultralytics import YOLO

model = YOLO('yolo11x')

result = model.predict('input_files/input_video.mp4', save = True)

print(result)

print("Boundary Boxes:")
for box in result[0].boxes:
    print(box)