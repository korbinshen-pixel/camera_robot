#!/usr/bin/env python3
"""
robot_controller.py — 差速行驶键盘控制节点
=============================================
差速行驶原理：
  左轮速度 v_L = v - ω * d/2
  右轮速度 v_R = v + ω * d/2
  其中 v = 线速度, ω = 角速度, d = 轮距

插件 libgazebo_ros_diff_drive 接收 Twist 消息后自动换算左右轮速，
本节点只需发布合适的 (linear.x, angular.z) 组合即可实现差速行驶。

键位说明：
  W        — 加速前进（线速度 +）
  S        — 加速后退（线速度 -）
  A        — 增大左转角速度（边走边转）
  D        — 增大右转角速度（边走边转）
  Q        — 原地左转（线速度=0，仅角速度）
  E        — 原地右转（线速度=0，仅角速度）
  空格/X   — 紧急停止（速度清零）
  Z        — 减速（线速度向 0 靠近）
  Ctrl+C   — 退出

速度特性：
  - 线速度、角速度均支持累加，模拟惯性感
  - 超过最大值自动钳位
  - 松键后自动线性衰减（可配置）
"""

import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ────────────────────────── 速度参数 ──────────────────────────
MAX_LINEAR   = 1.5    # m/s   最大线速度
MAX_ANGULAR  = 2.5    # rad/s 最大角速度
STEP_LINEAR  = 0.08   # m/s   每次按键线速度增量
STEP_ANGULAR = 0.15   # rad/s 每次按键角速度增量
DECAY_LINEAR = 0.85   # 每个控制周期线速度衰减系数（松键时）
DECAY_ANGULAR= 0.75   # 每个控制周期角速度衰减系数（松键时）
CTRL_HZ      = 20     # 控制频率 (Hz)

BANNER = """
┌─────────────────────────────────────────┐
│       差速四轮小车  键盘控制节点         │
├─────────────────────────────────────────┤
│  W / S   前进 / 后退（累加线速度）       │
│  A / D   左转 / 右转（边走边转）         │
│  Q / E   原地左转 / 原地右转             │
│  Z       减速                           │
│  空格/X  紧急停止                       │
│  Ctrl+C  退出                           │
├─────────────────────────────────────────┤
│  差速原理：v_L = v - ω·d/2             │
│            v_R = v + ω·d/2             │
│  轮距 d = 0.38 m（由插件参数决定）      │
└─────────────────────────────────────────┘
"""

KEY_BINDINGS = {
    'w': ( STEP_LINEAR,   0.0),
    's': (-STEP_LINEAR,   0.0),
    'a': ( 0.0,           STEP_ANGULAR),
    'd': ( 0.0,          -STEP_ANGULAR),
    'q': (-MAX_LINEAR,    STEP_ANGULAR * 3),   # 原地左转：清线速、大角速
    'e': (-MAX_LINEAR,   -STEP_ANGULAR * 3),   # 原地右转
    'z': ( 0.0,           0.0),                # 减速（特殊处理）
    ' ': None,                                  # 紧急停止
    'x': None,                                  # 紧急停止
}


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        self.pub = self.create_publisher(Twist, '/camera_robot/cmd_vel', 10)

        self.v = 0.0   # 当前线速度
        self.w = 0.0   # 当前角速度

        # 保存终端原始设置
        self._settings = termios.tcgetattr(sys.stdin)

        self.get_logger().info(BANNER)

    # ── 读取单个按键（非阻塞，超时 1/CTRL_HZ 秒） ──
    def _get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 1.0 / CTRL_HZ)
        key = sys.stdin.read(1) if rlist else ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._settings)
        return key

    # ── 发布当前速度 ──
    def _publish(self):
        msg = Twist()
        msg.linear.x  = self.v
        msg.angular.z = self.w
        self.pub.publish(msg)

        # 计算等效左右轮速（仅用于显示，轮距 0.38 m）
        d = 0.38
        v_l = self.v - self.w * d / 2
        v_r = self.v + self.w * d / 2
        self.get_logger().info(
            f"v={self.v:+.2f} m/s  ω={self.w:+.2f} rad/s  "
            f"| 左轮={v_l:+.2f} m/s  右轮={v_r:+.2f} m/s",
            throttle_duration_sec=0.2
        )

    # ── 主循环 ──
    def run(self):
        prev_key_active = False   # 上一拍是否有有效按键

        try:
            while rclpy.ok():
                key = self._get_key()

                if key == '\x03':   # Ctrl+C
                    break

                key_active = False

                if key in (' ', 'x'):
                    # 紧急停止
                    self.v = 0.0
                    self.w = 0.0
                    key_active = True

                elif key == 'z':
                    # 主动减速
                    self.v *= 0.5
                    self.w *= 0.5
                    key_active = True

                elif key in ('q', 'e'):
                    # 原地转：先将线速度归零，再施加角速度
                    self.v = 0.0
                    _, dw = KEY_BINDINGS[key]
                    self.w = clamp(self.w + dw, -MAX_ANGULAR, MAX_ANGULAR)
                    key_active = True

                elif key in KEY_BINDINGS:
                    dv, dw = KEY_BINDINGS[key]
                    self.v = clamp(self.v + dv, -MAX_LINEAR,  MAX_LINEAR)
                    self.w = clamp(self.w + dw, -MAX_ANGULAR, MAX_ANGULAR)
                    key_active = True

                else:
                    # 无按键：速度自然衰减（模拟摩擦）
                    self.v *= DECAY_LINEAR
                    self.w *= DECAY_ANGULAR
                    # 速度极小时直接清零，避免漂移
                    if abs(self.v) < 0.01:
                        self.v = 0.0
                    if abs(self.w) < 0.01:
                        self.w = 0.0

                self._publish()
                rclpy.spin_once(self, timeout_sec=0.0)

        except Exception as e:
            self.get_logger().error(f"控制循环异常: {e}")
        finally:
            # 停车
            self.v = 0.0
            self.w = 0.0
            self._publish()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._settings)


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
