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

## 3번

파일: turtlebot_patrol.xml

문제: 시뮬레이션 초기부터 장애물 감지됨.

원인:
라이다 방향이 의도(+Y)와는 반대(-Y)로 설정됨

해결
```xml
<site name="lidar_site" pos="0.07 0 0" quat="0.707 0 -0.707 0" size="0.005"/>
                                ↓
<site name="lidar_site" pos="0.07 0 0" quat="0.707 0 0.707 0" size="0.005"/>
```

---

## 4번

파일: patrol_env.xml

문제: 충전 스테이션에서 내려올 때 로봇이 회전을 함.

원인:
로봇이 충전 스테이션에서 내려올 때 바닥을 장애물로 감지함.

임시 해결:
충전 스테이션 바닥을 제거.

---

## 5번

파일: industrial_factory.xml

에러:
ValueError: Error: mass and inertia of moving bodies must be larger than mjMINVAL
Element name 'forklift', id 4, line 0

원인: forklift body(움직이는 body)의 질량 또는 관성이 0으로 계산되었다

임시 해결: freejoint 제거. 이후 단계별 추가.