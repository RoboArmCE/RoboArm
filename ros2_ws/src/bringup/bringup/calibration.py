import os
import cv2
import time
import yaml
import rclpy
import queue
import threading
import numpy as np
import sdk.common as common
from rclpy.node import Node
from dt_apriltags import Detector
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, CameraInfo 
from kinematics_msgs.srv import SetJointValue, GetRobotPose
from kinematics.kinematics_control import set_joint_value_target
from servo_controller_msgs.msg import ServosPosition
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from servo_controller.bus_servo_control import set_servo_position

class CalibrationNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.calibration_step = 0
        self.tags = []
        self.pose = []
        self.tag_count = 0
        self.K = None
        self.lock = threading.RLock()
        self.thread = None
        self.err_msg = None
        self.imgpts = None
        self.tag_size = 0.025
        self.tag_id = 1
        self.tag_id_2 = 100
        self.config_file = 'config.yaml'
        self.config_path = "/home/jetson/repos/ros2_ws/src/bringup/config/"
        
        # 创建发布者
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
        self.result_image_pub = self.create_publisher(Image, '~/image_result', 10)

        # 订阅
        self.image_sub = None
        self.camera_info_sub = None

        # 服务
        self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.create_service(Trigger, '~/start', self.start_calibration_srv_callback)

        self.at_detector = Detector(searchpath=['apriltags'],
                       families='tag36h11',
                       nthreads=4,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

        with open(self.config_path + self.config_file, 'r') as f:
            config = yaml.safe_load(f)
            # self.get_logger().info(config)

            # 转换为 numpy 数组
            self.extristric = np.array(config['extristric'])
            hand2cam_tf_matrix = np.array(config['hand2cam_tf_matrix'])
            white_area_pose_cam = np.array(config['white_area_pose_cam'])
            white_area_pose_cam = np.array(config['white_area_pose_cam'])
            white_area_pose_world = np.array(config['white_area_pose_world'])
            white_area_world_size = np.array(config['white_area_world_size'])



    def calibration_proc(self):
        self.tags = []
        set_servo_position(self.joints_pub, 2.0, ((1, 500), (2, 560), (3, 130), (4, 115), (5, 500), (10, 200)))  # 设置机械臂初始位置
        # actions.go_home(self.servos_pub)
        time.sleep(2)

        # 手眼标定。。。。(hand-eye calibration)
        # 通过固定的结构尺寸确定(calibration through fixed structural dimensions)

        # 获取当前末端坐标(get the current end-effector coordinates)
        # endpoint = rospy.ServiceProxy('/kinematics/get_current_pose', GetRobotPose)().pose 
        timer_cb_group = ReentrantCallbackGroup()
        self.set_joint_value_target_client = self.create_client(SetJointValue, '/kinematics/set_joint_value_target', callback_group=timer_cb_group)
        self.set_joint_value_target_client.wait_for_service()
        msg = set_joint_value_target([500, 560, 130, 115, 500])
        endpoint = self.send_request(self.set_joint_value_target_client, msg)
        pose_t = endpoint.pose.position
        pose_r = endpoint.pose.orientation
        

        self.endpoint = common.xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])
            
        # 获取标签数据(get tag data)
        t = time.time()
        self.tags = []
        self.calibration_step = 1
        while self.calibration_step == 1 and time.time() - t < 10:
            time.sleep(0.1)

        if len(self.tags) < 30:
            self.err_msg = "Time out, calibrate failed!!!"
            time.sleep(3)
            self.err_msg = None
            self.calibration_step = 0
            self.thread = None
            self.tags = []
            return

        # 读取 YAML 配置文件
        with open(self.config_path + self.config_file, 'r') as f:
            config = yaml.safe_load(f)

            # 转换为 numpy 数组
            hand2cam_tf_matrix = np.array(config['hand2cam_tf_matrix'])
            white_area_pose_cam = np.array(config['white_area_pose_cam'])
            white_area_pose_world = np.array(config['white_area_pose_world'])
        # 识别区域中心位置标定(calibration of the center position in the recognition area)
        # 对多次识别的数据求均值(calculate the average of multiple recognition data)
        pose = map(lambda tag: common.xyz_rot_to_mat(tag.pose_t, tag.pose_R), self.tags) # 将所有位姿转为4x4齐次矩阵(convert all poses to 4x4 homogeneous matrices)
        vectors = map(lambda p: p.ravel(), pose) # 将矩阵展平为向量(flatten the matrix into a vector)
        avg_pose = np.mean(list(vectors), axis=0).reshape((4, 4))  # 求均值并重组为4x4矩阵(calculate the mean and reassemble into a 4x4 matrix)
        pose_end = np.matmul(hand2cam_tf_matrix, avg_pose)  # 转换到末端相对坐标(transform to end-effector relative coordinates)

        pose_world = np.matmul(self.endpoint, pose_end)  # 转换到机械臂世界坐标(transform to robotic arm world coordinates)
        white_area_pose_cam= avg_pose.tolist()  # 识别区域中心的在相机的世界坐标系中的位置, 结果存入到param中(the position of the center of the recognition area in the camera's world coordinate system is stored in the parameter 'param')
        xyz, euler = common.mat_to_xyz_euler(avg_pose)
        white_area_pose_world = pose_world.tolist()  # 识别区域中心的机械臂世界坐标系的位置, 结果存入到param中(the position of the center of the recognition area in the camera's world coordinate system is stored in the parameter 'param')
        axyz, aeuler = common.mat_to_xyz_euler(pose_world)
        print(xyz, euler, axyz, aeuler)

        # 外参标定(extrinsic calibration)
        world_points = np.array([(-self.tag_size/2, -self.tag_size/2, 0), 
                                 ( self.tag_size/2, -self.tag_size/2, 0), 
                                 ( self.tag_size/2,  self.tag_size/2, 0), 
                                 (-self.tag_size/2,  self.tag_size/2, 0)] * len(self.tags), dtype=np.float64)

        image_points = np.array(list(map(lambda tag: tag.corners, self.tags)), dtype=np.float64).reshape((-1, 2))
        retval, rvec, tvec = cv2.solvePnP(world_points, image_points, self.K, self.D)
        rmat, _ = cv2.Rodrigues(rvec)
        self.extristric = tvec, rmat
        # self.get_logger().info(f"tvec: {tvec}, rmat: {rmat}")
        tvec_flattened = tvec.flatten().tolist()

        extristric = [
            tvec_flattened,  # 将 tvec 作为第一行
            rmat[0].tolist(),  # 第一行的旋转矩阵
            rmat[1].tolist(),  # 第二行的旋转矩阵
            rmat[2].tolist()   # 第三行的旋转矩阵
        ]
        # extristric = np.vstack((tvec.reshape((3, 1)), rmat)).tolist()# 外参存入param中(the extrinsic parameters are stored in 'param')

        # extristric= [tvec.reshape((3)).tolist(), rmat.reshape((-1, 3)).tolist()] # 外参存入param中(the extrinsic parameters are stored in 'param')
    
        data = {
            'white_area_pose_cam': white_area_pose_cam,
            'white_area_pose_world': white_area_pose_world,
            'extristric': extristric,
        }
        self.update_yaml_data(data, self.config_path + self.config_file)
     
        set_servo_position(self.joints_pub, 2.0, ((1, 500), (2, 560), (3, 130), (4, 115), (5, 500), (10, 200)))  # 设置机械臂初始位置
        self.calibration_step = 20
        time.sleep(3)
        self.calibration_step = 0
        self.thread = None

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()
    def update_yaml_data(self, new_data, yaml_file):
        # Step 1: Check if the file exists, and read the existing data
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)  # Load existing YAML data
        else:
            data = {}  # If file doesn't exist, create a new data dictionary

        # Step 2: Update the data with new data
        data.update(new_data)  # Update existing data with new data

        # Step 3: Write the updated data back to the YAML file
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f)
    
        time.sleep(0.1)
    # def save_yaml_data(self, data, yaml_file):
        # with open(yaml_file, 'w', encoding='utf-8') as f:
            # yaml.dump(data, f)
        # time.sleep(0.1)

    def start_calibration_srv_callback(self, request, response):
        # rospy.loginfo("开始进行标定")
        self.get_logger().info('\033[1;32m%s\033[0m' % "开始进行标定")
        with self.lock:
            if self.image_sub is None:
                err_msg = "Please call enter service first"
                self.get_logger().info(str(e))
                response.success = False
                response.message = "stop"
                return response  
            if self.thread is None:
                self.thread = threading.Thread(target=self.calibration_proc)
                self.thread.start()
                response.success = True
                response.message = "start"
                return response

    # def TriggerRequest(self, _: Trigger):
    def enter_srv_callback(self,request, response):
        # 获取和发布图像的topic(get and publish topic of image)
        self.image_sub = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/depth/camera_info', self.camera_info_callback, 1)
        response.success = True
        response.message = "start"
        return response
    def draw_retangle(self):
        # 读取 YAML 配置文件
        with open(self.config_path + self.config_file, 'r') as f:
            config = yaml.safe_load(f)

            # 转换为 numpy 数组
            extristric = np.array(config['extristric'])
            hand2cam_tf_matrix = np.array(config['hand2cam_tf_matrix'])
            white_area_pose_cam = np.array(config['white_area_pose_cam'])
            white_area_pose_cam = np.array(config['white_area_pose_cam'])
            white_area_pose_world = np.array(config['white_area_pose_world'])
            white_area_world_size = config['white_area_world_size']  # 保持为字典

            # 直接访问字典中的高度和宽度
            white_area_height = white_area_world_size['height']
            white_area_width = white_area_world_size['width']

        # 识别区域的四个角的世界坐标(the world coordinates of the four corners of the recognition area)
        # self.get_logger().info(f"extristric: {extristric}, shape: {extristric.shape}")
        white_area_cam = white_area_pose_cam
        white_area_center = white_area_pose_world
        self.white_area_center = white_area_center
        self.white_area_cam = white_area_cam
        white_area_center = white_area_center.reshape(4, 4)
        white_area_pose_cam = white_area_pose_cam.reshape(4, 4)
        white_area_cam = white_area_center.reshape(4, 4)
        # self.get_logger().info("white_area_center : " + str(white_area_center))
        euler_matrix = common.xyz_euler_to_mat((white_area_height / 2, white_area_width / 2 + 0.0, 0.0), (0, 0, 0))
        # self.get_logger().info("Euler matrix shape:"+str(euler_matrix.shape))
        white_area_lt = np.matmul(white_area_center, common.xyz_euler_to_mat((white_area_height / 2, white_area_width / 2 + 0.0, 0.0), (0, 0, 0)))
        white_area_lb = np.matmul(white_area_center, common.xyz_euler_to_mat((-white_area_height / 2, white_area_width / 2 + 0.0, 0.0), (0, 0, 0)))
        white_area_rb = np.matmul(white_area_center, common.xyz_euler_to_mat((-white_area_height / 2, -white_area_width / 2 -0.0, 0.0), (0, 0, 0)))
        white_area_rt = np.matmul(white_area_center, common.xyz_euler_to_mat((white_area_height / 2, -white_area_width / 2 -0.0, 0.0), (0, 0, 0)))

        msg = set_joint_value_target([500, 560, 130, 115, 500])
        endpoint = self.send_request(self.set_joint_value_target_client, msg)
        pose_t = endpoint.pose.position
        pose_r = endpoint.pose.orientation

        self.endpoint = common.xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])
        self.get_logger().info("self.endpoint shape: " + str(self.endpoint.shape))
        self.get_logger().info("hand2cam_tf_matrix shape:" + str(hand2cam_tf_matrix.shape))
        self.get_logger().info("hite_area_cam shape: " + str(white_area_cam.shape))
        corners_cam =  np.matmul(np.linalg.inv(np.matmul(self.endpoint, hand2cam_tf_matrix)), [white_area_lt, white_area_lb, white_area_rb, white_area_rt, white_area_center])
        corners_cam = np.matmul(np.linalg.inv(white_area_cam), corners_cam)
        corners_cam = corners_cam[:, :3, 3:].reshape((-1, 3))
        tvec = extristric[0]  # 取第一行
        rmat = extristric[1:]  # 取后面三行
        # tvec, rmat = extristric
        # tvec = [0.014433700240526136, 0.0070866308938068515, 0.1936181971273093]
  # 取第一行
        # rmat = [
    # [-0.03413093959864755, 0.996822685313855, 0.0719695283143447],
    # [0.9898784213890405, 0.04364296664247658, -0.13504074322583393],
    # [-0.1377526400126522, 0.06663201562551989, -0.9882228416016425]
# ]
  # 取后面三行

        # self.get_logger().info("extristric:"+ str(extristric))
        # self.get_logger().info("tvec:"+ str(tvec))
        # self.get_logger().info("rmat:"+ str(rmat))

        while self.K is None or self.D is None:
            time.sleep(0.5)

        center_imgpts, jac = cv2.projectPoints(corners_cam[-1:], np.array(rmat), np.array(tvec), self.K, self.D)
        self.center_imgpts = np.int32(center_imgpts).reshape(2)

        # self.get_logger().info("\032[91m" + "tvec: " + str(tvec) + "\032[0m")
        # self.get_logger().info("\032[91m" + "rmat: " + str(rmat) + "\032[0m")
        tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.030)
        self.get_logger().info("\033[91m" + "tvec: " + str(tvec) + "\033[0m")
        self.get_logger().info("\033[91m" + "rmat: " + str(rmat) + "\033[0m")

        imgpts, jac = cv2.projectPoints(corners_cam[:-1], np.array(rmat), np.array(tvec), self.K, self.D)
        self.get_logger().info("\033[90m" + "self.K: " + str(self.K) + "\033[0m")
        self.get_logger().info("\033[90m" + "self.D: " + str(self.D) + "\033[0m")
        # self.get_logger().info("tvec:"+ str(tvec))
        # self.get_logger().info("\033[91m" + "tvec: " + str(tvec) + "\033[0m")
        # self.get_logger().info("\033[91m" + "rmat: " + str(rmat) + "\033[0m")
        # self.get_logger().info("rmat:"+ str(rmat))
        self.imgpts = np.int32(imgpts).reshape(-1, 2)
        # self.get_logger().info("imgpts shape: " + str(self.imgpts))
        self.get_logger().info("imgpt : " + str(imgpts))


    def camera_info_callback(self, msg):
        with self.lock:
            K = np.matrix(msg.k).reshape(1, -1, 3)
            D = np.array(msg.d)
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (640, 480), 0, (640, 480))
            self.K, self.D = np.matrix(new_K), np.zeros((5, 1))
            # self.K = np.array([[453.94708252, 0, 327.56445312],
                   # [0, 453.94708252, 242.61904907],
                   # [0, 0, 1]])

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "stop calibration")

        response.success = True
        response.message = "stop"
        return response  
    

    def image_callback(self, ros_image):
        # 将ros格式图像转换为opencv格式(convert the ros format image to opencv format)
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
        result_image = np.copy(rgb_image)
        if self.K is not None:
            tags = self.at_detector.detect(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY), True, (self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]), self.tag_size)
            result_image = common.draw_tags(result_image, tags)
            if self.calibration_step == 1:
                if len(tags) == 1 and (tags[0].tag_id == self.tag_id or tags[0].tag_id == self.tag_id_2):
                    self.err_msg = None
                    if len(self.tags) > 0:
                        if common.distance(self.tags[-1].pose_t, tags[0].pose_t) < 0.003:
                            self.tags.append(tags[0])
                        else:
                            self.tags = []
                    else:
                        self.tags.append(tags[0])
                    if len(self.tags) >= 50:
                        print("收集完成")
                        self.calibration_step = 2
                else:
                    self.tags = []
                    if self.err_msg is None:
                        self.err_msg = "Please make sure there is only one tag in the;screen and the tag id is 1 or 100"
            if self.extristric is not None:
                world_points = np.array([(-self.tag_size/2, -self.tag_size/2, 0), 
                                         ( self.tag_size/2, -self.tag_size/2, 0), 
                                         ( self.tag_size/2,  self.tag_size/2, 0), 
                                         (-self.tag_size/2,  self.tag_size/2, 0)], dtype=np.float64)
                image_points, _ = cv2.projectPoints(world_points, self.extristric[1], self.extristric[0], self.K, self.D)
                image_points = image_points.astype(np.int32).reshape((-1, 2)).tolist()
                for p in image_points:
                    cv2.circle(result_image, tuple(p), 3, (0, 0, 0), -1)

        

        if self.err_msg is not None:
            self.get_logger().info(str(self.err_msg))
            err_msg = self.err_msg.split(';')
            for i, m in enumerate(err_msg):
                cv2.putText(result_image, m, (5, 50 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 6)
                cv2.putText(result_image, m, (5, 50 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        if self.calibration_step != 0:
            if self.calibration_step == 20:
                msg = "Calibration finished!"
                self.draw_retangle()
                # cv2.drawContours(result_image, [self.imgpts], -1, (255, 255, 0), 2, cv2.LINE_AA) # 绘制矩形(draw rectangle)
            else:
                msg = "Calibrating..."
            cv2.putText(result_image, msg, (5, result_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 6)
            cv2.putText(result_image, msg, (5, result_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        cv2.drawContours(result_image, [self.imgpts], -1, (255, 255, 0), 2, cv2.LINE_AA) # 绘制矩形(draw rectangle)
        # 计算帧率及发布结果图像(calculate the frame rate and publish the resulting image)
        ros_image.data = result_image.tobytes()
        self.result_image_pub.publish(ros_image)


def main():
    node = CalibrationNode('calibration')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()
