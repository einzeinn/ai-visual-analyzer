from ultralytics import YOLO
import cv2

# load model
model = YOLO("models/yolov8n.pt")

def detect_objects(image, threshold=0.5):

    results = model(image, conf=threshold)

    # ambil hasil gambar dengan bounding box
    annotated_frame = results[0].plot()

    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        if conf >= threshold:
            detections.append({
                "label": label,
                "confidence": round(conf * 100, 2),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

    return annotated_frame, detections