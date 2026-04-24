#!/usr/bin/env python3
"""
goal_navigator.py
=================
发布一个目标点，小车自动导航到达。

用法（另开终端）：
  ros2 topic pub --once /camera_robot/goal geometry_msgs/msg/Point "{x: 2.0, y: 1.0, z: 0.0}"

话题：
  订阅  /camera_robot/goal    (geometry_msgs/Point)   目标点
  订阅  /camera_robot/odom    (nav_msgs/Odometry)      当前位姿
  发布  /camera_robot/cmd_vel (geometry_msgs/Twist)    速度指令
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry


# ────────────────────── 参数 ──────────────────────
LINEAR_SPEED    = 0.4    # m/s   直行最大速度
ANGULAR_SPEED   = 0.8    # rad/s 转向最大速度
GOAL_TOLERANCE  = 0.15   # m     到达判定半径
ANGLE_TOLERANCE = 0.08   # rad   角度对齐容差（约 4.5°）
ANGULAR_KP      = 1.8    # 角度比例增益
LINEAR_KP       = 0.6    # 距离比例增益


class GoalNavigator(Node):
    def __init__(self):
        super().__init__('goal_navigator')

        # 当前位姿
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0

        # 目标点
        self.goal_x  = None
        self.goal_y  = None
        self.reached = True   # 初始无目标，不发速度

        # 订阅目标点
        self.goal_sub = self.create_subscription(
            Point,
            '/camera_robot/goal',
            self.goal_callback,
            10
        )

        # 订阅里程计
        self.odom_sub = self.create_subscription(
            Odometry,
            '/camera_robot/odom',
            self.odom_callback,
            10
        )

        # 发布速度
        self.cmd_pub = self.create_publisher(
            Twist,
            '/camera_robot/cmd_vel',
            10
        )

        # 控制定时器，20 Hz
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("目标点导航节点已启动！")
        self.get_logger().info(
            "发布目标：ros2 topic pub --once /camera_robot/goal "
            "geometry_msgs/msg/Point \"{x: 2.0, y: 1.0, z: 0.0}\""
        )

    # ────────────── 回调 ──────────────

    def goal_callback(self, msg: Point):
        self.goal_x  = msg.x
        self.goal_y  = msg.y
        self.reached = False   # 只有收到新目标才解锁，防止噪声触发
        self.get_logger().info(
            f"收到目标点：({self.goal_x:.2f}, {self.goal_y:.2f})"
        )

    def odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # 四元数 → yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw   = math.atan2(siny_cosp, cosy_cosp)

    # ────────────── 控制主循环 ──────────────

    def control_loop(self):
        # 已到达或无目标：持续发零速确保刹停，不做任何其他判断
        if self.goal_x is None or self.reached:
            self.stop()
            return

        dx       = self.goal_x - self.x
        dy       = self.goal_y - self.y
        distance = math.hypot(dx, dy)

        # ① 到达判定
        if distance < GOAL_TOLERANCE:
            self.stop()
            self.reached = True
            self.get_logger().info(
                f"已到达目标点！当前位置：({self.x:.2f}, {self.y:.2f})"
            )
            return

        # ② 计算目标方向与角度误差
        target_angle = math.atan2(dy, dx)
        angle_error  = self._normalize_angle(target_angle - self.yaw)

        cmd = Twist()

        # ③ 分阶段控制
        if abs(angle_error) > ANGLE_TOLERANCE:
            # 转向阶段：原地旋转对齐目标方向
            cmd.linear.x  = 0.0
            cmd.angular.z = max(-ANGULAR_SPEED,
                                min(ANGULAR_SPEED, ANGULAR_KP * angle_error))
        else:
            # 直行阶段：比例线速度 + 微小角速度修正
            # 保留 0.05 m/s 最低速，防止距离极小时速度归零造成停不进容差圈
            cmd.linear.x  = max(0.05,
                                min(LINEAR_SPEED, LINEAR_KP * distance))
            cmd.angular.z = max(-ANGULAR_SPEED * 0.3,
                                min(ANGULAR_SPEED * 0.3, ANGULAR_KP * angle_error))

        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"dist={distance:.2f}m  "
            f"angle_err={math.degrees(angle_error):.1f}°  "
            f"v={cmd.linear.x:.2f}  ω={cmd.angular.z:.2f}",
            throttle_duration_sec=0.5
        )

    def stop(self):
        """发送零速，物理刹停"""
        self.cmd_pub.publish(Twist())

    @staticmethod
    def _normalize_angle(angle):
        """将角度归一化到 (-π, π]"""
        while angle >  math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = GoalNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()