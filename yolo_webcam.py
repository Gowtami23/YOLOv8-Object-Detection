from ultralytics import YOLO
import cv2
import time
import os

# ----------------------------
# PARAMETERS (modify if needed)
# ----------------------------
MODEL_PATH = "yolov8n.pt"      # YOLOv8 model
MAX_DURATION = 20              # seconds before automatic stop
SAVE_VIDEO = True              # Set False if you don't want to save webcam video
OUTPUT_VIDEO = "webcam_output.mp4"  # Output video file name
FPS_DISPLAY = True             # Show FPS on screen
# ----------------------------

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Open webcam using DirectShow (Windows fix)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open webcam. Make sure no other app is using it!")
    exit()

# Get webcam properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)
if fps_input == 0:
    fps_input = 30  # default if webcam doesn't report FPS

# Set up video writer if needed
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_input, (width, height))

# Track time and FPS
start_time = time.time()
frame_count = 0

print("Webcam opened successfully. Press 'q' to stop manually.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)

    # Annotate frame
    annotated_frame = frame.copy()
    for r in results:
        annotated_frame = r.plot()

    # Calculate and display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    if FPS_DISPLAY:
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLOv8 Live Webcam Detection", annotated_frame)

    # Save to output video if needed
    if SAVE_VIDEO:
        out.write(annotated_frame)

    # Stop conditions
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user pressing q")
        break
    if time.time() - start_time > MAX_DURATION:
        print(f"Stopped automatically after {MAX_DURATION} seconds")
        break

# Release everything
cap.release()
if SAVE_VIDEO:
    out.release()
cv2.destroyAllWindows()
print(f"Webcam session ended. Output video saved as: {OUTPUT_VIDEO}" if SAVE_VIDEO else "Webcam session ended.")
