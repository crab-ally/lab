import cv2
import mediapipe as mp

# ==========================
# MediaPipe 모듈 설정
# ==========================

# BlazeFace 기반 얼굴 검출 모듈
mp_face = mp.solutions.face_detection

# Face Mesh (468개 얼굴 랜드마크 추출)
mp_mesh = mp.solutions.face_mesh

# 카메라 입력
cap = cv2.VideoCapture(0)

# ==========================
# 모델 초기화
# ==========================

# BlazeFace 얼굴 검출기
with mp_face.FaceDetection(
    model_selection=0,  # 0: Short Range (~2m), 1: Full Range (~5m)
    min_detection_confidence=0.5  # 얼굴 검출 신뢰도 threshold
) as detector:

    # Face Mesh 모델
    with mp_mesh.FaceMesh(
        max_num_faces=1,  # 검출할 최대 얼굴 개수
        refine_landmarks=True,  # 눈 주변 등 세밀한 landmark 사용
        min_detection_confidence=0.5,  # 최초 얼굴 검출 기준
        min_tracking_confidence=0.5  # 추적 신뢰도 기준
    ) as face_mesh:

        while True:

            # ==========================
            # 카메라 프레임 획득
            # ==========================

            ret, frame = cap.read()

            if not ret:
                break

            # 이미지 크기
            h, w, _ = frame.shape

            # OpenCV: BGR, MediaPipe: RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ==========================
            # 1. BlazeFace 얼굴 검출
            # ==========================

            result = detector.process(rgb)

            if result.detections:

                for detection in result.detections:

                    # 얼굴 Bounding Box 정보
                    # xmin, ymin, width, height는 이미지 크기 대비 0~1 비율
                    bbox = detection.location_data.relative_bounding_box

                    # ==========================
                    # Bounding Box Pixel 변환
                    # ==========================

                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)

                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    # ==========================
                    # Bounding Box Margin 추가
                    # ==========================

                    margin_x = 0.1   # 좌우 10%
                    margin_y = 0.1   # 상하 10%

                    # 좌상단 위치를 확장 방향으로 이동
                    x = int(x - width * margin_x)
                    y = int(y - height * margin_y)

                    # 크기 확장
                    width = int(width * (1 + 2 * margin_x))
                    height = int(height * (1 + 2 * margin_y))

                    # 이미지 영역 밖 제한
                    x = max(0, x)
                    y = max(0, y)

                    width = min(width, w - x)
                    height = min(height, h - y)

                    # 확장된 얼굴 영역 표시
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + width, y + height),
                        (0, 255, 0),
                        2
                    )

                    # ==========================
                    # 2. 얼굴 영역 Crop
                    # ==========================

                    # Margin이 적용된 얼굴 영역
                    face_img = rgb[
                        y:y + height,
                        x:x + width
                    ]

                    if face_img.size == 0:
                        continue

                    # ==========================
                    # 3. Face Mesh 실행
                    # ==========================
                    
                    mesh_result = face_mesh.process(face_img)

                    if mesh_result.multi_face_landmarks:

                        for face_landmarks in mesh_result.multi_face_landmarks:

                            # 사용할 랜드마크
                            #
                            # Head Pose
                            # 1   : 코 끝
                            # 152 : 턱 끝
                            # 33  : 왼쪽 눈 바깥
                            # 263 : 오른쪽 눈 바깥
                            # 61  : 왼쪽 입꼬리
                            # 291 : 오른쪽 입꼬리
                            #
                            # Eye 상태
                            # 133 : 왼쪽 눈 안쪽
                            # 159 : 왼쪽 눈 위
                            # 145 : 왼쪽 눈 아래
                            # 362 : 오른쪽 눈 안쪽
                            # 387 : 오른쪽 눈 위
                            # 374 : 오른쪽 눈 아래
                            landmark_indices = [
                                1, 152,
                                33, 263,
                                61, 291,
                                133, 159, 145,
                                362, 387, 374
                            ]

                            for idx in landmark_indices:

                                landmark = face_landmarks.landmark[idx]

                                # Face Mesh 좌표는 Crop 영역 기준
                                # 0~1 비율 좌표
                                #
                                # 원본 Frame 좌표로 변환
                                px = int(landmark.x * width + x)
                                py = int(landmark.y * height + y)

                                # 랜드마크 표시
                                cv2.circle(
                                    frame,
                                    (px, py),
                                    3,
                                    (0, 255, 0),
                                    -1
                                )

            # ==========================
            # 결과 출력
            # ==========================

            cv2.imshow(
                "BlazeFace + Face Mesh",
                frame
            )

            # ESC 종료
            if cv2.waitKey(1) == 27:
                break

# ==========================
# 종료 처리
# ==========================

cap.release()
cv2.destroyAllWindows()