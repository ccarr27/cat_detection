from ultralytics import YOLO
import cv2

# Load YOLOv5s model
model = YOLO("yolov5su.pt")  # PyTorch weights

cap = cv2.VideoCapture("IMG_7281.mov")
# cap = cv2.VideoCapture(0)

# Confidence threshold for detecting cats
CONF_THRESHOLD = 0.2  # try lower if cats are small
# You can also adjust input resolution for small objects inside YOLOv5 config if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, imgsz=640)  # imgsz can be 640 or 1280 for better small-object detection

    # Draw results
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if r.names[cls] == "cat" and conf > CONF_THRESHOLD:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"cat {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Cat Detection - YOLOv5", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()