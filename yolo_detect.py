from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Input and output folders
image_folder = "images"
output_folder = "processed_images"
os.makedirs(output_folder, exist_ok=True)

# Store processed images for display
processed_images = []

# Loop through all images
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    print(f"Processing: {img_path}")

    results = model(img_path)

    for r in results:
        img = r.plot()  # Draw detections on the image

        # Save the processed image
        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, img)
        print(f"Saved processed image to: {save_path}")

        # Store for display
        processed_images.append((img_name, img))

# Display all images in separate windows
for name, img in processed_images:
    cv2.imshow(name, img)

# Wait until a key is pressed to close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
