# 에러 정정표

## 1번

파일: scripts\spawn_world.py

기존 코드
```py
print(f"[경고] 장애물 감지! 거리: {dist:.2f}m - 우회 경로 탐색 중...")
```

에러: TypeError: unsupported format string passed to numpy.ndarray.__format__

원인: 
data.sensor(...).data는 NumPy 배열을 반환한다. 그리고 .2f는 실수를 출력하는 포맷팅이다.

해결
```py
lidar_dist = data.sensor('lidar').data # [numpy.ndarray]
                ↓
lidar_dist = data.sensor('lidar').data[0] # [numpy.float64]
```

---

## 2번

파일: scripts\spawn_world.py

기존 코드
```py
if lidar_dist < 1.0:
```

에러: 장애물이 없음에도 장애물 감지됨.

원인: 
장애물 미감지 시 `-1`을 반환하며, 이것이 if문의 조건을 만족함.

해결
```py
if lidar_dist < 1.0:
        ↓
if lidar_dist >= 0 and lidar_dist < 1.0
```

---