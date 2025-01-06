from ultralytics import YOLO
import cv2
import shutil  # For clearing directories

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Define the output directory
save_dir = "runs/detect/predict"  # Use a unique or fixed directory

# Clear the save directory before prediction to avoid conflicts
shutil.rmtree(save_dir, ignore_errors=True)  # Remove previous contents

# Predict on the new image
detection_output = model.predict(source=r"C:\Users\gs705\Desktop\Image Detection\img\1.jpg", conf=0.25, save=True, save_dir=save_dir)

# Path to the newly saved image
saved_image_path = f"{save_dir}/1.jpg"  # Adjust this based on YOLO's naming convention

# Load the saved image
detected_image = cv2.imread(saved_image_path)

# Display the image
cv2.imshow("Detected Image", detected_image)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()  # Close the image window

# Display tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].numpy())
