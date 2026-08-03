import cv2
import mediapipe as mp
import math

mp_mesh = mp.solutions.face_mesh

EAR_THRESHOLD = 0.27

# [좌, 우, 상1, 상2, 하1, 하2]
LEFT_EYE_IDX = [33, 133, 159, 158, 145, 153]
RIGHT_EYE_IDX = [362, 263, 386, 385, 374, 380]


def calculate_ear(eye_points):
    vertical1 = math.dist(eye_points[2], eye_points[4])
    vertical2 = math.dist(eye_points[3], eye_points[5])
    horizontal = math.dist(eye_points[0], eye_points[1])

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


cap = cv2.VideoCapture(0)

with mp_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,      # DMS에는 홍채 불필요
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:

            landmarks = result.multi_face_landmarks[0]

            left_eye = [
                (
                    int(landmarks.landmark[i].x * w),
                    int(landmarks.landmark[i].y * h)
                )
                for i in LEFT_EYE_IDX
            ]

            right_eye = [
                (
                    int(landmarks.landmark[i].x * w),
                    int(landmarks.landmark[i].y * h)
                )
                for i in RIGHT_EYE_IDX
            ]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            ear = (left_ear + right_ear) / 2

            status = "EYES CLOSED" if ear < EAR_THRESHOLD else "EYES OPEN"
            color = (0, 0, 255) if ear < EAR_THRESHOLD else (0, 255, 0)

            cv2.putText(
                frame,
                f"L:{left_ear:.3f}  R:{right_ear:.3f}  EAR:{ear:.3f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                status,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            # 랜드마크 표시
            for p in left_eye + right_eye:
                cv2.circle(frame, p, 2, (255, 0, 0), -1)

            # EAR 계산에 사용되는 선 표시
            for eye in [left_eye, right_eye]:
                cv2.line(frame, eye[0], eye[1], (0, 255, 255), 1)  # 수평
                cv2.line(frame, eye[2], eye[4], (0, 255, 255), 1)  # 수직1
                cv2.line(frame, eye[3], eye[5], (0, 255, 255), 1)  # 수직2

        cv2.imshow("DMS Eye Detection", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()