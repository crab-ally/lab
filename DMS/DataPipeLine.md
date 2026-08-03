```
                 차량 내부
                    |
                    v
              IR Camera / RGB Camera
                    |
                    v
              OpenCV Frame 획득
                    |
                    v
        +----------------------------+
        |       Face Detection       |
        |        (BlazeFace)         |
        +----------------------------+
                    |
                    |
        얼굴 위치 Bounding Box
        (x, y, width, height)
                    |
                    v
        Bounding Box Margin 추가
        (상하좌우 여유 영역)
                    |
                    v
              Face Crop
              얼굴 영역 추출
                    |
                    v
        +----------------------------+
        |        Face Mesh            |
        |   468 Facial Landmark       |
        +----------------------------+
                    |
                    v
          Landmark 좌표 추출
                    |
        +-----------+-------------+
        |                         |
        v                         v
   눈 랜드마크                  얼굴 주요점
        |                         |
        v                         v
      EAR 계산               Head Pose 계산
        |                         |
        v                         v
   눈 감김 판단              고개 방향 판단
        |                         |
        +------------+------------+
                     |
                     v
              운전자 상태 판단
                     |
          +----------+----------+
          |                     |
          v                     v
       졸음 감지           전방주시 태만
```