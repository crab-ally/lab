import cv2
import mediapipe as mp

# MediaPipe Face Detection (BlazeFace)
mp_face = mp.solutions.face_detection

# 카메라 입력
cap = cv2.VideoCapture(0)

# BlazeFace 초기화
with mp_face.FaceDetection(
    model_selection=0,  # 0: short-range(~2m), 1: long-range(~5m)
    min_detection_confidence=0.5
) as detector:

    while True:

        # 카메라 프레임 획득
        ret, frame = cap.read()

        if not ret:
            break

        # 이미지 크기
        h, w, _ = frame.shape

        # OpenCV: BGR, MediaPipe: RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # BlazeFace 얼굴 검출
        result = detector.process(rgb)

        if result.detections:

            for detection in result.detections:

                # 얼굴 Bounding Box
                # xmin, ymin, width, height는 0~1 비율 좌표
                bbox = detection.location_data.relative_bounding_box

                # 비율 좌표 -> Pixel 좌표 변환
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)

                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # ==========================
                # Bounding Box Margin 추가
                # ==========================

                # 좌우 10%, 상하 10% 확장
                margin_x = 0.1
                margin_y = 0.1

                # 좌상단 위치를 바깥쪽으로 이동
                x = int(x - width * margin_x)
                y = int(y - height * margin_y)

                # 전체 크기 증가
                width = int(width * (1 + 2 * margin_x))
                height = int(height * (1 + 2 * margin_y))


                # 이미지 영역 밖으로 나가지 않도록 제한
                x = max(0, x)
                y = max(0, y)

                width = min(width, w - x)
                height = min(height, h - y)


                # 확장된 얼굴 Bounding Box 표시
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2
                )

        # 결과 출력
        cv2.imshow(
            "BlazeFace",
            frame
        )

        # ESC 종료
        if cv2.waitKey(1) == 27:
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()