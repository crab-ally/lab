from ultralytics import YOLO
import cv2

# ==========================
# 1. 학습된 모델 불러오기
# ==========================
model = YOLO("/workspace/models/ppe_yolov8n/best.pt")

# ==========================
# 2. 학습에 사용했던 이미지
# ==========================
image_path = "/workspace/datasets/ppe_dataset/images/train/000093.jpg"

# ==========================
# 3. 추론
# ==========================
results = model.predict(
    source=image_path,
    imgsz=640,
    conf=0.05,
    save=False,
    verbose=False
)

# ==========================
# 4. 결과 출력
# ==========================
img = cv2.imread(image_path)

for r in results:
    print(f"Detected: {len(r.boxes)} objects")

    for box in r.boxes:
        cls = int(box.cls)
        conf = float(box.conf)

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"{model.names[cls]} {conf:.2f}"

        print(label)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

# ==========================
# 5. 화면 출력
# ==========================
cv2.imshow("Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()