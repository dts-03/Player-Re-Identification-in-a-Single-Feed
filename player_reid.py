import cv2
from ultralytics import YOLO
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv11 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('best.pt')
model.to(device)

# Initialize video
cap = cv2.VideoCapture('15sec_input_720p.mp4')

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

frame_id = 0
show_display = True  # Set to False to benchmark speed without display
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % 2 != 0:  # Process every other frame for speed
        continue

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (640, 360))

    # Run YOLO detection
    results = model(frame_resized)
    detections = []
    result = results[0]  # Get the first result for this frame
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
        confs = result.boxes.conf.cpu().numpy()  # (N, 1)
        clss = result.boxes.cls.cpu().numpy()    # (N, 1)
        for box, conf, cls in zip(boxes, confs, clss):
            if int(cls) == 0:  # assuming class 0 is player
                x1, y1, x2, y2 = box
                detections.append(([float(x1), float(y1), float(x2-x1), float(y2-y1)], float(conf), "player"))

    # Update tracker
    detections = [d for d in detections if isinstance(d, tuple) and len(d) == 3 and isinstance(d[0], list) and len(d[0]) == 4]
    if len(detections) == 0:
        tracks = []
    else:
        tracks = tracker.update_tracks(detections, frame=frame_resized)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        cv2.rectangle(frame_resized, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0,255,0), 2)
        cv2.putText(frame_resized, f'ID: {track_id}', (int(ltrb[0]), int(ltrb[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Show or save frame
    if show_display:
        cv2.imshow('Player Re-ID', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows() 