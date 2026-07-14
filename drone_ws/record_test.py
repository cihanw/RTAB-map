#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Import from world_config to get the correct topic dynamically
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src/drone_sim/launch'))
from world_config import WORLD_NAME
CAMERA_TOPIC = f'/world/{WORLD_NAME}/model/drone/model/d455/link/link/sensor/realsense_d455/image'

class FrameRecorder(Node):
    def __init__(self):
        super().__init__('frame_recorder')
        self.subscription = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        self.frame_count = 0
        self.save_interval = 60
        
        self.save_dir = os.path.join(os.getcwd(), 'log', 'frames')
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f'Frame recorder started. Saving 1 in {self.save_interval} frames to {self.save_dir}')

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.save_interval == 0:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                filename = os.path.join(self.save_dir, f'frame_{self.frame_count:06d}.jpg')
                cv2.imwrite(filename, cv_image)
                self.get_logger().info(f'Saved {filename}')
            except Exception as e:
                self.get_logger().error(f'Error saving frame: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = FrameRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
