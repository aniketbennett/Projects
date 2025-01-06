import random
import cv2
import numpy as np
from ultralytics import YOLO

# List of classes (add class names manually if needed)
class_list = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
]

# Generate random colors for classes
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(class_list))]

# Load a pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")

# Set video capture
cap = cv2.VideoCapture(r"C:\Users\gs705\Desktop\New folder\video_sample2.mp4")

if not cap.isOpened():
    print("Cannot open video file")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Run YOLO detection
    detect_params = model.predict(source=frame, conf=0.45, save=False)

    # Process detections
    for result in detect_params:
        boxes = result.boxes  # Get bounding boxes
        for box in boxes:
            clsID = int(box.cls.numpy()[0])  # Class ID
            conf = box.conf.numpy()[0]  # Confidence
            bb = box.xyxy.numpy()[0]  # Bounding box coordinates (x_min, y_min, x_max, y_max)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[clsID],
                2,
            )

            # Display class name and confidence
            cv2.putText(
                frame,
                f"{class_list[clsID]} {conf:.2f}",
                (int(bb[0]), int(bb[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Terminate when "Q" is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
