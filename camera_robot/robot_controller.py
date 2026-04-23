#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import select
import termios
import tty

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # 创建发布者
        self.publisher_ = self.create_publisher(Twist, '/camera_robot/cmd_vel', 10)
        
        # 设置速度参数
        self.linear_speed = 0.5
        self.angular_speed = 1.0
        
        self.get_logger().info("键盘控制节点已启动!")
        self.get_logger().info("使用 WASD 控制机器人移动:")
        self.get_logger().info("  W: 前进")
        self.get_logger().info("  S: 后退")
        self.get_logger().info("  A: 左转")
        self.get_logger().info("  D: 右转")
        self.get_logger().info("  Q: 停止")
        self.get_logger().info("  Ctrl+C: 退出")
        
        # 保存终端设置
        self.settings = termios.tcgetattr(sys.stdin)
        
    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
    
    def run(self):
        try:
            while rclpy.ok():
                key = self.get_key()
                
                if key == 'w':
                    self.move_forward()
                elif key == 's':
                    self.move_backward()
                elif key == 'a':
                    self.turn_left()
                elif key == 'd':
                    self.turn_right()
                elif key == 'q':
                    self.stop()
                elif key == '\x03':  # Ctrl+C
                    break
                    
                rclpy.spin_once(self, timeout_sec=0.1)
                
        except Exception as e:
            self.get_logger().error(f"发生错误: {e}")
        finally:
            # 恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
    
    def move_forward(self):
        msg = Twist()
        msg.linear.x = self.linear_speed
        self.publisher_.publish(msg)
        self.get_logger().info("前进")
    
    def move_backward(self):
        msg = Twist()
        msg.linear.x = -self.linear_speed
        self.publisher_.publish(msg)
        self.get_logger().info("后退")
    
    def turn_left(self):
        msg = Twist()
        msg.angular.z = self.angular_speed
        self.publisher_.publish(msg)
        self.get_logger().info("左转")
    
    def turn_right(self):
        msg = Twist()
        msg.angular.z = -self.angular_speed
        self.publisher_.publish(msg)
        self.get_logger().info("右转")
    
    def stop(self):
        msg = Twist()
        self.publisher_.publish(msg)
        self.get_logger().info("停止")

def main(args=None):
    rclpy.init(args=args)
    
    controller = RobotController()
    
    try:
        controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()