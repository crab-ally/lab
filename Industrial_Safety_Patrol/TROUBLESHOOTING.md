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