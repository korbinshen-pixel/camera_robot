#!/usr/bin/env python3
"""
lidar_viewer.py
激光雷达监视节点：订阅 /camera_robot/scan，实时打印统计信息并
可选发布一张简易俯视图到 /camera_robot/lidar_image。

话题：
  订阅  /camera_robot/scan          (sensor_msgs/LaserScan)
  发布  /camera_robot/lidar_image   (sensor_msgs/Image, BGR8, 300×300)
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class LidarViewer(Node):
    def __init__(self):
        super().__init__('lidar_viewer')

        self.bridge = CvBridge()

        # ---------- 订阅者 ----------
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/camera_robot/scan',
            self.scan_callback,
            10
        )

        # ---------- 发布者（俯视图） ----------
        self.image_pub = self.create_publisher(
            Image,
            '/camera_robot/lidar_image',
            10
        )

        # 俯视图参数
        self.img_size = 400          # 像素
        self.max_range_vis = 6.0     # 显示的最大距离（米）

        self.get_logger().info("激光雷达监视节点已启动！")
        self.get_logger().info("  订阅：/camera_robot/scan")
        self.get_logger().info("  发布俯视图：/camera_robot/lidar_image")

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)

        # ---------- 统计信息 ----------
        valid = ranges[np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)]
        if valid.size > 0:
            self.get_logger().info(
                f"[LiDAR] points={valid.size}/{len(ranges)}  "
                f"min={valid.min():.2f}m  max={valid.max():.2f}m  "
                f"mean={valid.mean():.2f}m"
            )
        else:
            self.get_logger().info("[LiDAR] 无有效点")

        # ---------- 生成俯视图 ----------
        canvas = self._make_canvas(msg, ranges)
        img_msg = self.bridge.cv2_to_imgmsg(canvas, "bgr8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "lidar_link"
        self.image_pub.publish(img_msg)

    def _make_canvas(self, msg: LaserScan, ranges: np.ndarray):
        """生成 400×400 的激光雷达俯视图。"""
        size = self.img_size
        scale = (size / 2) / self.max_range_vis   # 像素/米
        center = (size // 2, size // 2)

        canvas = np.zeros((size, size, 3), dtype=np.uint8)

        # 绘制同心圆（距离刻度）
        for r_m in [1, 2, 3, 4, 5, 6]:
            r_px = int(r_m * scale)
            if r_px < size // 2:
                cv2.circle(canvas, center, r_px, (40, 40, 40), 1)
                cv2.putText(canvas, f"{r_m}m",
                            (center[0] + r_px + 2, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

        # 绘制十字准星
        cv2.line(canvas, (0, center[1]), (size, center[1]), (50, 50, 50), 1)
        cv2.line(canvas, (center[0], 0), (center[0], size), (50, 50, 50), 1)

        # 绘制激光点
        angle = msg.angle_min
        for r in ranges:
            if math.isfinite(r) and msg.range_min < r < min(msg.range_max, self.max_range_vis):
                px = int(center[0] + r * scale * math.cos(angle))
                py = int(center[1] - r * scale * math.sin(angle))  # 图像 y 轴向下
                if 0 <= px < size and 0 <= py < size:
                    # 颜色按距离从绿→红渐变
                    ratio = r / self.max_range_vis
                    color = (0, int(255 * (1 - ratio)), int(255 * ratio))
                    cv2.circle(canvas, (px, py), 2, color, -1)
            angle += msg.angle_increment

        # 机器人自身（黄色圆点）
        cv2.circle(canvas, center, 5, (0, 220, 220), -1)
        cv2.putText(canvas, "Robot", (center[0] + 7, center[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 220), 1)

        # 标题
        cv2.putText(canvas, "LiDAR Top-View", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        return canvas


def main(args=None):
    rclpy.init(args=args)
    node = LidarViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
