# 에러 정정표 (Troubleshooting Log)

cobot pnp 프로젝트 진행 중 발생한 오류와 해결 기록

---

# 1. 외부 모델 <compiler> 덮어쓰기(Override)에 따른 Asset 경로 유실 에러

## 파일

`scene/panda_test.xml`, `world/test.xml`

## 에러코드

```py
ValueError: Error: Error opening file 'D:\my_work/lab/cobot_pnp_safety/mujoco_menagerie/franka_emika_panda/link4.stl'
```

## 원인

최상위 panda_test.xml에 작성한 <compiler> 태그가 원본 panda.xml 내부의 <compiler meshdir="assets"/> 설정을 강제로 덮어씌움(Override). 이로 인해 STL 메쉬 탐색 경로에서 assets/ 디렉터리가 탈락하고, franka_emika_panda/link4.stl이라는 잘못된 경로로 접근하여 파일 로딩 실패.

## 해결

world/test.xml ㅡ Include 구문 지원을 위해 최상위를 <mujocoinclude> 태그로 작성하고 배경 요소만 배치

```xml
<mujocoinclude>
  <light pos="0 0 3" dir="0 0 -1" directional="true"/>
  <geom name="floor" type="plane" size="3 3 0.1" material="groundplane"/>
</mujocoinclude>
```

scene/panda_test.xml ㅡ strippath="true"를 주면 panda.xml 내부에서 요구하는 assets/link4.stl 경로의 앞부분(assets/)을 자동으로 떼어내고 파일명(link4.stl)만 추출하여, 지정된 meshdir(.../franka_emika_panda/assets)에서 정확하게 메쉬를 찾아냅니다.

```xml
<mujoco model="panda_scene">
  <!-- 덮어쓰기 문제 해결: meshdir을 assets로 명시하고 strippath="true" 적용 -->
  <compiler angle="radian" meshdir="../mujoco_menagerie/franka_emika_panda/assets" strippath="true"/>

  <statistic center="0 0 0.5" extent="2.0"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba fog="0.2 0.3 0.4 1"/>
  </visual>

  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" 
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" 
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texrepeat="5 5" texuniform="true"/>
  </asset>

  <worldbody>
    <!-- 배경 include -->
    <include file="../world/test.xml"/>
  </worldbody>

  <!-- 원본 판다 include (worldbody 밖 최상위) -->
  <include file="../mujoco_menagerie/franka_emika_panda/panda.xml"/>
</mujoco>
```

---

# 2. XML Include 중첩 구조 및 VFS Asset 경로 매핑 에러

## 파일

- `scene/panda_test.xml`
- `world/test.xml`
- `scripts/mujoco_viewer.py`

## 에러코드

```py
ValueError: Error: Error opening file 'D:\my_work/lab/cobot_pnp_safety/model/franka_emika_panda/link0.stl'
ValueError: Error: Error opening file 'assets/link0.stl'
ValueError: Repeated file name in assets dict: finger_0.obj
```

## 원인

1. 상대 경로 탐색 한계 및 Compiler Override

scene/panda_test.xml에서 하위 model/franka_emika_panda/panda.xml을 include할 때, 상위 XML의 <compiler> 설정이 하위 XML의 <compiler meshdir="assets"/> 속성을 강제로 덮어씌워(Override) 메쉬 탐색 경로에서 assets/ 디렉터리가 유실됨.

2. XML 스키마 규칙 위반

world/test.xml을 루트 태그 없이 include하거나 최상위 위치에 두어 <light>, <geom> 등이 올바른 부모 노드(<worldbody>) 없이 로드되어 파싱 에러 발생.

3. VFS 키 중복 (Key Collision)

파이썬 VFS(Virtual File System) 매핑 시 파일 이름 단독(file_path.name)을 키로 사용할 경우, 서로 다른 폴더에 위치한 동명의 에셋(finger_0.obj 등) 간의 충돌 발생.

## 해결

world/test.xml ㅡ Include 구문 지원을 위해 최상위 루트를 <mujocoinclude> 태그로 작성하고 배경 요소만 배치

```xml
<mujocoinclude>
  <light name="world_light" pos="0 0 3" dir="0 0 -1" directional="true"/>
  <geom name="floor" type="plane" size="3 3 0.1" material="groundplane"/>
</mujocoinclude>
```

scene/panda_test.xml ㅡ 최상위 XML에서는 기본 환경 설정만 유지하고, 배경 요소는 <worldbody> 내부로, 로봇 모델은 최상위 레벨로 각각 include.

```xml
<mujoco model="panda_scene">
  <compiler angle="radian"/>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.5 0.5 0.5" specular="0.2 0.2 0.2"/>
  </visual>

  <asset>
    <texture name="ground_texture" type="2d" builtin="checker" rgb1="0.25 0.25 0.25" rgb2="0.12 0.12 0.12" mark="edge" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="ground_texture" texrepeat="5 5" texuniform="true"/>
  </asset>

  <worldbody>
    <include file="../world/test.xml"/>
  </worldbody>

  <include file="../model/franka_emika_panda/panda.xml"/>
</mujoco>
```

scripts/mujoco_viewer.py ㅡ MuJoCo C++ 로더의 상대 경로 탐색 한계를 우회하기 위해 파이썬 VFS 메모리 매핑 기법을 적용. 디렉터리 구조를 포함한 상대 경로(relative_to(MODEL_DIR))만 키로 등록하여 키 중복 문제 해결.

```xml
import os
import mujoco
import mujoco.viewer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANDA_XML = PROJECT_ROOT / "scene/panda_test.xml"
MODEL_DIR = PROJECT_ROOT / "model/franka_emika_panda"

def main():
    print(f"[LOAD] {PANDA_XML}")
    
    try:
        with open(PANDA_XML, "r", encoding="utf-8") as f:
            xml_string = f.read()

        # 상대 경로(assets/...) 기반 VFS 매핑으로 경로 충돌 및 유실 문제 해결
        vfs_assets = {}
        if MODEL_DIR.exists():
            for file_path in MODEL_DIR.rglob("*"):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(MODEL_DIR)).replace("\\", "/")
                    with open(file_path, "rb") as f:
                        vfs_assets[rel_path] = f.read()

        model = mujoco.MjModel.from_xml_string(xml_string, assets=vfs_assets)
        data = mujoco.MjData(model)

    except Exception as e:
        print(f"[ERROR] {e}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
```