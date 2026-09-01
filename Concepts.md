# Segmentation

이미지의 각 픽셀이 "무엇에 속하는지" 구분하는 것

1. Semantic Segmentation : 각 픽셀을 클래스 단위로 분류
  - 사람 → {class: 1}
  - 의자 → {class: 2}
  - 바닥 → {class: 3}
2. Instance Segmentation : 같은 클래스여도 개별 객체별로 구분
  - 사람 A → {instance: 1}
  - 사람 B → {instance: 2}
  - 의자 A → {instance: 3}

> 일반적 ㅡ AI가 객체 인식 & segmentation 수행  
> cobot_pnp_safety 활용 ㅡ mujoco가 geom id를 직접 알려줌

**segmentation mask**

```py
_PANDA_BODY_NAMES={
    "link0","link1","link2","link3",
    "link4","link5","link6","link7",
    "hand","left_finger","right_finger",
    "wrist_camera_link"
}
mask = np.isin(geom_id_map,panda_ids_arr) # 마스크 생성
depth_metric[robot_mask]=np.nan           # 마스크 사용
```

---