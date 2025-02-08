#!/usr/bin/env python3
# encoding: utf-8
#颜色识别
import cv2
import rclpy
from rclpy.node import Node
import math
import numpy as np
import sdk.common as common
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ColorDetection(Node):
    def __init__(self, name):
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.rgb_image = None
        self.bgr_image = None
        self.result_image = None
        self.bridge = CvBridge()  # 用于ROS Image消息与OpenCV图像之间的转换

        self.display = self.get_parameter('enable_display').value
        self.data = common.get_yaml_data("/home/jetson/repos/RoboArm/ros2_ws/src/app/config/lab_config.yaml")  
        self.lab_data = self.data['/**']['ros__parameters'] 


        # 订阅图像话题
        self.image_sub = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)

        #发布颜色识别处理图像
        self.image_pub = self.create_publisher(Image, '/color_detection/result_image', 1)

    # 处理ROS Image消息，将其转换为OpenCV图像
    def image_callback(self, ros_image):
        try:
            
            # 将 ROS Image 消息转换为 OpenCV 图像
            self.rgb_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")

            # 将图像的颜色空间转换成LAB
            self.lab_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2LAB)

            # RGB转BGR
            self.bgr_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)


            # 开始颜色检测
            self.color_detection("red",self.bgr_image)

            # 展示原图和检测结果
            if  self.result_image is not None:
                cv2.imshow('BGR', self.bgr_image)
                cv2.imshow('result_image', self.result_image)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error('Failed to convert image: {}'.format(e))

    # 腐蚀与膨胀操作
    def erode_and_dilate(self, binary, kernel=3):
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
        eroded = cv2.erode(binary, element)  # 腐蚀
        dilated = cv2.dilate(eroded, element)  # 膨胀
        return dilated

    # 颜色识别函数
    def color_detection(self, color, bgr_image):
        self.result_image = cv2.GaussianBlur(bgr_image, (3, 3), 3)
        self.result_image = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2LAB)
        # 设置颜色阈值并生成二值化图像
        self.result_image = cv2.inRange(self.result_image,
                                      np.array(self.lab_data['color_range_list'][color]['min']),
                                      np.array(self.lab_data['color_range_list'][color]['max']))
        self.result_image = self.erode_and_dilate(self.result_image)

        # 找出所有轮廓
        contours = cv2.findContours(self.result_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = math.fabs(cv2.contourArea(c))
            if area >= 2000:
                if self.display:
                    self.get_logger().info(f"Detected {color}")
        
        # 将二值化的图像转换为BGR格式，便于发布
        # result_image = cv2.cvtColor(self.result_image, cv2.COLOR_GRAY2BGR)

        # 发布彩色 BGR 图像
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.result_image, "mono8"))

   
def main():
    rclpy.init()
    node = ColorDetection('color_detection')
    rclpy.spin(node)
    camera_node.destroy_node()
    rclpy.shutdown()
        
if __name__ == '__main__':
    main()

