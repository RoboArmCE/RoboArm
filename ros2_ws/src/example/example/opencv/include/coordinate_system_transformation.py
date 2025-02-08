#!/usr/bin/env python3
# encoding: utf-8
#物体位姿计算
import cv2
import time
import rclpy
import numpy as np
from sdk import common
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as RosImage, CameraInfo
 


EXTRISTRIC = [0.013496478947862782, 0.007814062898814457, 0.1989498730143979, 
            0.014199741155075718, 0.9993175319866384, -0.03410040491347389,
            0.991079993351924, -0.018585856051489102, -0.13196595292871552,
            -0.1325096756040082, -0.03192234670207295, -0.9906675273028551]

WHITE_AREA_POSE_WORLD = [
      [-0.9984441214151412, -0.009066499920881304, 0.0548033910293186, 0.19013239192234654],
      [-0.011402967162491971, 0.9989682975914446, -0.042313780155303786, -0.013458563172451893],
      [-0.05437878102272995, -0.04287348168095714, -0.9975187212300515, 0.009979679994607485],
      [0.0, 0.0, 0.0, 1.0]
      ]

class CoordinateTransformation(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.bridge = CvBridge()  # 用于ROS Image消息与OpenCV图像之间的转换
        self.K = None
        rmat = EXTRISTRIC[3:]  # 旋转矩阵
        tvec = EXTRISTRIC[:3]  # 平移向量
        rmat = np.array(rmat).reshape(3, 3)
        tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.030)
        self.extristric = tvec, rmat
        white_area_center = WHITE_AREA_POSE_WORLD
        self.white_area_center = white_area_center

        # 订阅图像话题
        self.image_sub = self.create_subscription(RosImage, '/color_detection/result_image', self.image_callback, 1)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/depth/camera_info', self.camera_info_callback, 1)

    def camera_info_callback(self, msg):
        self.K = np.matrix(msg.k).reshape(1, -1, 3)

    # 处理ROS节点数据
    def image_callback(self, result_image):
        try:
            
            # 将 ROS Image 消息转换为 OpenCV 图像
            bgr_image = self.bridge.imgmsg_to_cv2(result_image, "bgr8")

            # 将灰度图像转换为 BGR 图像
            result_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)


            if result_image is not None :
                # 计算识别到的轮廓
                contours = cv2.findContours(result_image, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2]  # 找出所有轮廓

                if contours :

                    # 找出最大轮廓
                    c = max(contours, key = cv2.contourArea)
                    # 根据轮廓大小判断是否进行下一步处理
                    rect = cv2.minAreaRect(c)  # 获取最小外接矩形
                    corners = np.int0(cv2.boxPoints(rect))  # 获取最小外接矩形的四个角点
                    x, y = rect[0][0],rect[0][1]
                    # 打印像素坐标
                    self.get_logger().info(f"像素坐标为: x: {x}, y: {y}")


                    projection_matrix = np.row_stack((np.column_stack((self.extristric[1],self.extristric[0])), np.array([[0, 0, 0, 1]])))
                    world_pose = common.pixels_to_world([[x,y]], self.K, projection_matrix)[0]
                    world_pose[1] = -world_pose[1]
                    world_pose[2] = 0.03
                    world_pose = np.matmul(self.white_area_center, common.xyz_euler_to_mat(world_pose, (0, 0, 0)))
                    world_pose[2] = 0.03
                    pose_t, _ = common.mat_to_xyz_euler(world_pose)
                    self.get_logger().info(f"实际坐标为： {pose_t}")
                else:
                    self.get_logger().info("未检测到所需识别的颜色，请将色块放置到相机视野内。")

        except Exception as e:
            print(e)

def main():
    node = CoordinateTransformation('coordinate_transformation')
    rclpy.spin(node)
    camera_node.destroy_node()
    
    rclpy.shutdown()
        
if __name__ == '__main__':
    main()
