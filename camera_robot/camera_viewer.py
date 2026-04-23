#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        
        # 创建图像转换器
        self.bridge = CvBridge()
        
        # 创建订阅者
        self.subscription = self.create_subscription(
            Image,
            '/camera_robot/rgbd_camera/image_raw',
            self.image_callback,
            10
        )
        
        self.get_logger().info("相机显示节点已启动!")
    
    def image_callback(self, msg):
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 显示图像
            cv2.imshow('RGB Camera View', cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"图像转换错误: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    viewer = CameraViewer()
    
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        viewer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()