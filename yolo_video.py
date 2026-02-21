from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")
input_video = "videos/video.mp4"
output_video = "output_video.mp4"

cap = cv2.VideoCapture(input_video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for r in results:
        annotated_frame = r.plot()

    out.write(annotated_frame)
    cv2.imshow("YOLO Video Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved as: {output_video}")
