#!/usr/bin/env python3
"""
MuJoCo XML -> MoveIt 2 Planning Scene 자동 장애물 등록기

기능:
  - MuJoCo world/test.xml 자동 파싱
  - body의 pos / quat / euler 누적
  - box / sphere / cylinder 자동 변환
  - MoveIt Planning Scene에 CollisionObject 등록
  - /planning_scene 토픽 publish가 아니라
    /apply_planning_scene 서비스를 사용하여 직접 적용

현재 프로젝트 기준:
  XML: /workspace/world/test.xml
  MoveIt 기준 frame: link0

현재 world/test.xml:
  table1
    - table1_top
    - table1_leg1~4
  table2
    - table2_top
    - table2_leg1~4

제외:
  - floor
  - pnp_object
  - Panda robot 관련 body
  - camera body
  - light
"""

import os
import math
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive


class MujocoPlanningScene(Node):

    def __init__(self):
        super().__init__("mujoco_planning_scene")

        # ============================================================
        # 기본 설정
        # ============================================================

        self.world_xml = "/workspace/world/test.xml"
        self.frame_id = "link0"

        # ============================================================
        # 등록하지 않을 body / geom
        # ============================================================

        self.ignore_body_names = {
            "pnp_object",
            "ceiling_camera_link",
        }

        self.ignore_geom_names = {
            "floor",
        }

        # ============================================================
        # Panda robot 관련 이름
        # ============================================================

        self.robot_body_names = {
            "panda",
            "link0", "link1", "link2", "link3",
            "link4", "link5", "link6", "link7",
            "hand", "left_finger", "right_finger",
        }

        # ============================================================
        # ApplyPlanningScene 서비스
        # ============================================================

        self.apply_scene_client = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
        )

        self.get_logger().info("=" * 60)
        self.get_logger().info("MuJoCo World -> MoveIt Planning Scene")
        self.get_logger().info("Automatic Collision Object Loader")
        self.get_logger().info("=" * 60)

        self.load_and_apply_scene()

    # ================================================================
    # Vector parsing
    # ================================================================

    def parse_vec3(self, text, default=(0.0, 0.0, 0.0)):
        """
        XML의 pos / size / euler 등을 tuple로 변환한다.
        """
        if not text:
            return default

        values = text.split()
        if len(values) < 3:
            return default

        try:
            return tuple(map(float, values[:3]))
        except ValueError:
            return default

    # ================================================================
    # Quaternion
    # ================================================================

    def quat_multiply(self, q1, q2):
        """
        Quaternion 곱셈.
        내부 표현: (x, y, z, w)
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        return (
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        )

    def quat_to_matrix(self, q):
        """
        Quaternion을 rotation matrix로 변환.
        """
        x, y, z, w = q

        return [
            [
                1 - 2*(y*y + z*z),
                2*(x*y - z*w),
                2*(x*z + y*w),
            ],
            [
                2*(x*y + z*w),
                1 - 2*(x*x + z*z),
                2*(y*z - x*w),
            ],
            [
                2*(x*z - y*w),
                2*(y*z + x*w),
                1 - 2*(x*x + y*y),
            ],
        ]

    def rotate_vector(self, q, v):
        """
        Quaternion으로 local vector를 회전시킨다.
        """
        m = self.quat_to_matrix(q)
        x, y, z = v

        return (
            m[0][0]*x + m[0][1]*y + m[0][2]*z,
            m[1][0]*x + m[1][1]*y + m[1][2]*z,
            m[2][0]*x + m[2][1]*y + m[2][2]*z,
        )

    # ================================================================
    # Transform composition
    # ================================================================

    def compose_transform(
        self,
        parent_pos,
        parent_quat,
        local_pos,
        local_quat,
    ):
        """
        부모 transform과 자식 transform을 합친다.

        예:
          table1 pos=(0, 0.55, 0.4)
          geom   pos=(0.22, 0.14, -0.4)

        결과:
          (0.22, 0.69, 0.0)
        """
        rotated = self.rotate_vector(
            parent_quat,
            local_pos,
        )

        pos = tuple(
            parent_pos[i] + rotated[i]
            for i in range(3)
        )

        quat = self.quat_multiply(
            parent_quat,
            local_quat,
        )

        return pos, quat

    # ================================================================
    # MuJoCo quaternion
    # ================================================================

    def parse_quat(self, element):
        """
        MuJoCo quat="w x y z" -> ROS/internal (x y z w)
        """
        values = element.get("quat", "").split()

        if len(values) != 4:
            return 0.0, 0.0, 0.0, 1.0

        try:
            w, x, y, z = map(float, values)
        except ValueError:
            return 0.0, 0.0, 0.0, 1.0

        return x, y, z, w

    def euler_to_quat(self, euler):
        """
        Euler XYZ -> Quaternion.
        내부 표현: x y z w
        """
        rx, ry, rz = euler

        cx, sx = math.cos(rx/2), math.sin(rx/2)
        cy, sy = math.cos(ry/2), math.sin(ry/2)
        cz, sz = math.cos(rz/2), math.sin(rz/2)

        return (
            sx*cy*cz - cx*sy*sz,
            cx*sy*cz + sx*cy*sz,
            cx*cy*sz - sx*sy*cz,
            cx*cy*cz + sx*sy*sz,
        )

    def get_element_quaternion(self, element):
        """
        quat가 있으면 quat 사용.
        없고 euler가 있으면 euler 사용.
        둘 다 없으면 identity.
        """
        if element.get("quat") is not None:
            return self.parse_quat(element)

        if element.get("euler") is not None:
            return self.euler_to_quat(
                self.parse_vec3(element.get("euler"))
            )

        return 0.0, 0.0, 0.0, 1.0

    # ================================================================
    # Body transform
    # ================================================================

    def get_body_transform(self, body, parent_pos, parent_quat):
        """
        Body의 local transform을 parent transform과 합친다.
        """
        return self.compose_transform(
            parent_pos,
            parent_quat,
            self.parse_vec3(body.get("pos")),
            self.get_element_quaternion(body),
        )

    # ================================================================
    # CollisionObject 생성
    # ================================================================

    def create_collision_object(
        self,
        geom,
        body_name,
        body_pos,
        body_quat,
    ):
        """
        MuJoCo geom 하나를 MoveIt CollisionObject로 변환한다.
        """
        geom_name = geom.get(
            "name",
            f"{body_name}_geom",
        )

        if geom_name in self.ignore_geom_names:
            self.get_logger().info(
                f"[SKIP] Ignored geom: {geom_name}"
            )
            return None

        geom_type = geom.get("type", "")

        if geom_type not in ("box", "sphere", "cylinder"):
            self.get_logger().warn(
                f"[SKIP] Unsupported geom: "
                f"{geom_name}, type={geom_type}"
            )
            return None

        # ------------------------------------------------------------
        # geom local transform
        # ------------------------------------------------------------

        world_pos, world_quat = self.compose_transform(
            body_pos,
            body_quat,
            self.parse_vec3(geom.get("pos")),
            self.get_element_quaternion(geom),
        )

        collision = CollisionObject()
        collision.header.frame_id = self.frame_id
        collision.id = geom_name

        primitive = SolidPrimitive()
        pose = Pose()

        pose.position.x, pose.position.y, pose.position.z = world_pos
        pose.orientation.x = world_quat[0]
        pose.orientation.y = world_quat[1]
        pose.orientation.z = world_quat[2]
        pose.orientation.w = world_quat[3]

        # ============================================================
        # BOX
        # ============================================================

        if geom_type == "box":
            size = self.parse_vec3(geom.get("size"))

            # MuJoCo box size는 half-size
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [2.0 * s for s in size]

        # ============================================================
        # SPHERE
        # ============================================================

        elif geom_type == "sphere":
            values = geom.get("size", "").split()

            if not values:
                return None

            primitive.type = SolidPrimitive.SPHERE
            primitive.dimensions = [float(values[0])]

        # ============================================================
        # CYLINDER
        # ============================================================

        elif geom_type == "cylinder":
            size = self.parse_vec3(geom.get("size"))

            # MuJoCo:
            #   size[0] = radius
            #   size[1] = half-height
            primitive.type = SolidPrimitive.CYLINDER
            primitive.dimensions = [2.0 * size[1], size[0]]

        collision.primitives.append(primitive)
        collision.primitive_poses.append(pose)
        collision.operation = CollisionObject.ADD

        self.get_logger().info(
            f"[ADD] {geom_name} "
            f"type={geom_type} "
            f"pos=({world_pos[0]:.3f}, "
            f"{world_pos[1]:.3f}, "
            f"{world_pos[2]:.3f})"
        )

        return collision

    # ================================================================
    # Body recursive processing
    # ================================================================

    def process_body(
        self,
        body,
        parent_pos,
        parent_quat,
        collision_objects,
    ):
        """
        MuJoCo body를 재귀적으로 탐색한다.

        예:
          table1
            |
            +-- geom
            |
            +-- body
                  |
                  +-- geom
        """
        body_name = body.get("name", "unnamed_body")

        body_pos, body_quat = self.get_body_transform(
            body,
            parent_pos,
            parent_quat,
        )

        is_ignored = body_name in self.ignore_body_names
        is_robot = (
            body_name in self.robot_body_names
            or body_name.startswith("panda")
            or body_name.startswith("link")
        )

        # ------------------------------------------------------------
        # 현재 body의 geom 처리
        # ------------------------------------------------------------

        if not is_ignored and not is_robot:
            for geom in body.findall("geom"):
                collision = self.create_collision_object(
                    geom,
                    body_name,
                    body_pos,
                    body_quat,
                )

                if collision is not None:
                    collision_objects.append(collision)

        # ------------------------------------------------------------
        # Child body 재귀 처리
        # ------------------------------------------------------------

        for child in body.findall("body"):
            self.process_body(
                child,
                body_pos,
                body_quat,
                collision_objects,
            )

    # ================================================================
    # XML Load
    # ================================================================

    def load_collision_objects(self):
        """
        XML을 읽어서 CollisionObject 목록을 만든다.
        """
        if not os.path.exists(self.world_xml):
            self.get_logger().error(
                f"[XML] File not found: {self.world_xml}"
            )
            return []

        self.get_logger().info(
            f"[XML] Loading: {self.world_xml}"
        )

        try:
            root = ET.parse(self.world_xml).getroot()
        except Exception as e:
            self.get_logger().error(
                f"[XML] Parse failed: {e}"
            )
            return []

        world_pos = (0.0, 0.0, 0.0)
        world_quat = (0.0, 0.0, 0.0, 1.0)
        collision_objects = []

        # ------------------------------------------------------------
        # worldbody 찾기
        #
        # 일반 MJCF:
        #   <worldbody>
        #     <body ...>
        #
        # 현재 test.xml:
        #   <mujocoinclude>
        #     <body ...>
        # ------------------------------------------------------------

        worldbody = root.find("worldbody")

        bodies = (
            worldbody.findall("body")
            if worldbody is not None
            else root.findall("body")
        )

        for body in bodies:
            self.process_body(
                body,
                world_pos,
                world_quat,
                collision_objects,
            )

        return collision_objects

    # ================================================================
    # Apply Planning Scene
    # ================================================================

    def apply_collision_objects(self, collision_objects):
        """
        MoveIt의 /apply_planning_scene 서비스를 사용해서
        CollisionObject를 Planning Scene에 직접 적용한다.
        """
        if not collision_objects:
            self.get_logger().warn(
                "[Planning Scene] No collision objects."
            )
            return False

        self.get_logger().info(
            "[Planning Scene] "
            "Waiting for /apply_planning_scene ..."
        )

        if not self.apply_scene_client.wait_for_service(
            timeout_sec=10.0
        ):
            self.get_logger().error(
                "[Planning Scene] "
                "/apply_planning_scene unavailable."
            )
            return False

        # ------------------------------------------------------------
        # PlanningScene 생성
        # ------------------------------------------------------------

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = collision_objects

        request = ApplyPlanningScene.Request()
        request.scene = scene

        self.get_logger().info(
            f"[Planning Scene] Applying "
            f"{len(collision_objects)} collision objects..."
        )

        future = self.apply_scene_client.call_async(request)

        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

        if not future.done():
            self.get_logger().error(
                "[Planning Scene] "
                "Service call did not finish."
            )
            return False

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(
                f"[Planning Scene] Service exception: {e}"
            )
            return False

        if response is None:
            self.get_logger().error(
                "[Planning Scene] Empty service response."
            )
            return False

        if not response.success:
            self.get_logger().error(
                "[Planning Scene] "
                "ApplyPlanningScene returned success=False."
            )
            return False

        self.get_logger().info("=" * 60)
        self.get_logger().info("[Planning Scene] SUCCESS")
        self.get_logger().info(
            f"Registered {len(collision_objects)} "
            f"collision objects."
        )
        self.get_logger().info("=" * 60)

        return True

    # ================================================================
    # Load + Apply
    # ================================================================

    def load_and_apply_scene(self):
        """
        전체 작업:

          XML
           ↓
          parse
           ↓
          CollisionObject
           ↓
          ApplyPlanningScene
        """
        collision_objects = self.load_collision_objects()

        if not collision_objects:
            self.get_logger().error(
                "[Planning Scene] No objects generated."
            )
            return

        self.apply_collision_objects(collision_objects)


def main(args=None):
    rclpy.init(args=args)
    node = MujocoPlanningScene()

    try:
        # 서비스 적용 후 결과 확인 시간을 준다.
        rclpy.spin_once(node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()