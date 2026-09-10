#!/usr/bin/env python3
"""
MuJoCo XML -> MoveIt 2 Planning Scene 자동 장애물 등록기
(UR5e + Robotiq 2F-85)

MuJoCo WORLD 기준: UR5e base 위치 = (0, 0, 0.6)
MoveIt 기준 frame: base
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

        self.world_xml = "/workspace/world/table_bottle.xml"
        self.frame_id = "base"

        # MuJoCo world에서 UR5e base 원점
        self.base_world_pos = (0.0, 0.0, 0.6)

        # 등록 제외 object
        self.ignore_body_names = {
            "water_bottle",
            "ur5e_bottom_box",
        }
        self.ignore_geom_names = {"floor"}

        self.robot_body_names = {
            "base",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
            # Robotiq 2F-85 gripper
            "base_mount",
            "2f85_base",
            "right_driver",
            "right_coupler",
            "right_spring_link",
            "right_follower",
            "right_pad",
            "right_silicone_pad",
            "left_driver",
            "left_coupler",
            "left_spring_link",
            "left_follower",
            "left_pad",
            "left_silicone_pad",
        }

        self.apply_scene_client = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
        )

        self.load_and_apply_scene()

    # ================================================================
    # Basic math
    # ================================================================

    def vec3(self, text, default=(0.0, 0.0, 0.0)):
        if not text:
            return default
        try:
            v = list(map(float, text.split()[:3]))
            return tuple(v) if len(v) == 3 else default
        except ValueError:
            return default

    def quat_mul(self, a, b):
        x1, y1, z1, w1 = a
        x2, y2, z2, w2 = b
        return (
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        )

    def quat_matrix(self, q):
        x, y, z, w = q
        return [
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ]

    def rotate(self, q, v):
        m = self.quat_matrix(q)
        return tuple(
            sum(m[i][j] * v[j] for j in range(3))
            for i in range(3)
        )

    def quat(self, element):
        if element.get("quat"):
            try:
                w, x, y, z = map(
                    float, element.get("quat").split()
                )
                return x, y, z, w
            except ValueError:
                pass

        if element.get("euler"):
            rx, ry, rz = self.vec3(element.get("euler"))
            cx, sx = math.cos(rx/2), math.sin(rx/2)
            cy, sy = math.cos(ry/2), math.sin(ry/2)
            cz, sz = math.cos(rz/2), math.sin(rz/2)

            return (
                sx*cy*cz - cx*sy*sz,
                cx*sy*cz + sx*cy*sz,
                cx*cy*sz - sx*sy*cz,
                cx*cy*cz + sx*sy*sz,
            )

        return 0.0, 0.0, 0.0, 1.0

    def compose(self, p, q, lp, lq):
        rp = self.rotate(q, lp)
        return (
            tuple(p[i] + rp[i] for i in range(3)),
            self.quat_mul(q, lq),
        )

    # ================================================================
    # WORLD -> BASE
    # ================================================================

    def world_to_base(self, world_pos):
        return tuple(
            world_pos[i] - self.base_world_pos[i]
            for i in range(3)
        )

    # ================================================================
    # Body
    # ================================================================

    def body_transform(self, body, parent_pos, parent_quat):
        return self.compose(
            parent_pos,
            parent_quat,
            self.vec3(body.get("pos")),
            self.quat(body),
        )

    # ================================================================
    # Collision Object
    # ================================================================

    def create_collision(
        self,
        geom,
        body_name,
        body_pos,
        body_quat,
    ):
        name = geom.get(
            "name",
            f"{body_name}_geom",
        )

        if name in self.ignore_geom_names:
            return None

        gtype = geom.get("type")

        if gtype not in ("box", "sphere", "cylinder"):
            self.get_logger().warn(
                f"[SKIP] {name}: type={gtype}"
            )
            return None

        world_pos, world_quat = self.compose(
            body_pos,
            body_quat,
            self.vec3(geom.get("pos")),
            self.quat(geom),
        )

        base_pos = self.world_to_base(world_pos)

        collision = CollisionObject()
        collision.header.frame_id = self.frame_id
        collision.id = name

        primitive = SolidPrimitive()
        pose = Pose()

        pose.position.x, pose.position.y, pose.position.z = base_pos
        pose.orientation.x = world_quat[0]
        pose.orientation.y = world_quat[1]
        pose.orientation.z = world_quat[2]
        pose.orientation.w = world_quat[3]

        if gtype == "box":
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [
                2.0 * x for x in self.vec3(geom.get("size"))
            ]

        elif gtype == "sphere":
            size = geom.get("size", "").split()
            if not size:
                return None
            primitive.type = SolidPrimitive.SPHERE
            primitive.dimensions = [float(size[0])]

        else:
            size = self.vec3(geom.get("size"))
            primitive.type = SolidPrimitive.CYLINDER
            primitive.dimensions = [
                2.0 * size[1],
                size[0],
            ]

        collision.primitives.append(primitive)
        collision.primitive_poses.append(pose)
        collision.operation = CollisionObject.ADD

        self.get_logger().info(
            f"[ADD] {name} "
            f"world=({world_pos[0]:.3f}, "
            f"{world_pos[1]:.3f}, "
            f"{world_pos[2]:.3f}) "
            f"-> base=({base_pos[0]:.3f}, "
            f"{base_pos[1]:.3f}, "
            f"{base_pos[2]:.3f})"
        )

        return collision

    # ================================================================
    # Recursive body processing
    # ================================================================

    def process_body(
        self,
        body,
        parent_pos,
        parent_quat,
        collision_objects,
    ):
        name = body.get("name", "unnamed")

        body_pos, body_quat = self.body_transform(
            body,
            parent_pos,
            parent_quat,
        )

        ignored = name in self.ignore_body_names
        robot = (
            name in self.robot_body_names
            or name.startswith("ur5e")
            or name.startswith("2f85")
        )

        if not ignored and not robot:
            for geom in body.findall("geom"):
                collision = self.create_collision(
                    geom,
                    name,
                    body_pos,
                    body_quat,
                )
                if collision:
                    collision_objects.append(collision)

        for child in body.findall("body"):
            self.process_body(
                child,
                body_pos,
                body_quat,
                collision_objects,
            )

    # ================================================================
    # XML
    # ================================================================

    def load_collision_objects(self):
        if not os.path.exists(self.world_xml):
            self.get_logger().error(
                f"[XML] File not found: {self.world_xml}"
            )
            return []

        try:
            root = ET.parse(self.world_xml).getroot()
        except Exception as e:
            self.get_logger().error(
                f"[XML] Parse failed: {e}"
            )
            return []

        worldbody = root.find("worldbody")
        bodies = (
            worldbody.findall("body")
            if worldbody is not None
            else root.findall("body")
        )

        objects = []

        for body in bodies:
            self.process_body(
                body,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
                objects,
            )

        return objects

    # ================================================================
    # Apply
    # ================================================================

    def apply_collision_objects(self, objects):
        if not objects:
            self.get_logger().error(
                "[Planning Scene] No collision objects."
            )
            return False

        if not self.apply_scene_client.wait_for_service(
            timeout_sec=10.0
        ):
            self.get_logger().error(
                "[Planning Scene] Service unavailable."
            )
            return False

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = objects

        request = ApplyPlanningScene.Request()
        request.scene = scene

        future = self.apply_scene_client.call_async(request)

        while rclpy.ok() and not future.done():
            rclpy.spin_once(
                self,
                timeout_sec=0.1,
            )

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(str(e))
            return False

        if not response or not response.success:
            self.get_logger().error(
                "[Planning Scene] Apply failed."
            )
            return False

        self.get_logger().info(
            f"[Planning Scene] SUCCESS: "
            f"{len(objects)} objects, frame={self.frame_id}"
        )

        return True

    # ================================================================
    # Main
    # ================================================================

    def load_and_apply_scene(self):
        objects = self.load_collision_objects()

        if objects:
            self.apply_collision_objects(objects)


def main(args=None):
    rclpy.init(args=args)
    node = MujocoPlanningScene()

    try:
        rclpy.spin_once(node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()