import time
from collections import Counter

import cv2
from ultralytics import YOLO

CONF_THRESHOLD = 0.5
WINDOW_NAME = "Deteksi Orang dan Objek"

COLOR_PERSON = (0, 200, 0)
COLOR_OBJECT = (0, 140, 255)
COLOR_FACE = (255, 200, 0)
COLOR_NONE = (0, 0, 255)


def load_models():
    detector = YOLO("yolov8n.pt")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Model deteksi wajah gagal dimuat")
    return detector, face_cascade


def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
    )


def detect_objects(frame, detector):
    result = detector.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        category = "Orang" if label == "person" else "Objek"
        detections.append((x1, y1, x2, y2, label, category, float(box.conf[0])))
    return detections


def draw_box(frame, x1, y1, x2, y2, text, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    top = max(y1 - th - 8, 0)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 8, top + th + 8), color, -1)
    cv2.putText(
        frame, text, (x1 + 4, top + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )


def build_summary(detections, faces):
    persons = sum(1 for d in detections if d[5] == "Orang")
    objects = Counter(d[4] for d in detections if d[5] == "Objek")

    if persons == 0 and not objects:
        return "Tidak ada orang atau objek terdeteksi", COLOR_NONE

    parts = []
    if persons:
        face_note = f", {len(faces)} wajah terlihat" if len(faces) else ", wajah tidak terlihat"
        parts.append(f"Orang: {persons}{face_note}")
    if objects:
        parts.append("Objek: " + ", ".join(f"{k} x{v}" for k, v in objects.items()))
    return " | ".join(parts), COLOR_PERSON if persons else COLOR_OBJECT


def main():
    detector, face_cascade = load_models()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Kamera tidak dapat dibuka")

    previous_time = time.time()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Gagal membaca frame dari kamera")
                break

            detections = detect_objects(frame, detector)
            faces = detect_faces(frame, face_cascade)

            for x1, y1, x2, y2, label, category, conf in detections:
                color = COLOR_PERSON if category == "Orang" else COLOR_OBJECT
                draw_box(frame, x1, y1, x2, y2, f"{category}: {label} {conf:.0%}", color)

            for x, y, w, h in faces:
                draw_box(frame, x, y, x + w, y + h, "Wajah", COLOR_FACE)

            summary, summary_color = build_summary(detections, faces)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (0, 0, 0), -1)
            cv2.putText(
                frame, summary, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, summary_color, 2, cv2.LINE_AA,
            )

            now = time.time()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()