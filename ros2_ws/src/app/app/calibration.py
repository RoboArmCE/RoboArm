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
from app.common import Heart
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
from dt_apriltags import Detector
from sensor_msgs.msg import Image, CameraInfo 
from kinematics_msgs.srv import GetRobotPose
from kinematics.kinematics_control import set_joint_value_target
from servo_controller_msgs.msg import ServosPosition
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from servo_controller.bus_servo_control import set_servo_position

class CalibrationNode(Node):
    hand2cam_tf_matrix = [
    [0.0, 0.0, 1.0, -0.101],
    [-1.0, 0.0, 0.0, 0.011],
    [0.0, -1.0, 0.0, 0.045],
    [0.0, 0.0, 0.0, 1.0]
]
    
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.calibration_step = 0
        self.tags = []
        self.pose = []
        self.tag_count = 0
        self.K = None
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
        self.err_msg = None
        self.imgpts = None
        self.thread_started = False 
        self.tag_size = 0.025
        self.tag_id = 1
        self.tag_id_2 = 100
        self.config_file = 'transform.yaml'
        self.config_path = "/home/enrique/repos/ros2_ws/src/app/config/"
        self.white_area_width = 0.175
        self.white_area_height = 0.135
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)
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

            # 转换为 numpy 数组
            self.extristric = np.array(config['extristric'])
            self.white_area_pose_cam = np.array(config['white_area_pose_cam'])
            self.white_area_pose_world = np.array(config['white_area_pose_world'])
        Heart(self, '~/heartbeat', 5, lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response()))  # 心跳包(heartbeat package)

    def calibration_proc(self):
        self.tags = []
        set_servo_position(self.joints_pub, 2.0, ((1, 500), (2, 560), (3, 130), (4, 115), (5, 500), (10, 200)))  # 设置机械臂初始位置

        # 手眼标定。。。。(hand-eye calibration)
        # 通过固定的结构尺寸确定(calibration through fixed structural dimensions)

        # 获取当前末端坐标(get the current end-effector coordinates)
        timer_cb_group = ReentrantCallbackGroup()
        self.get_current_pose_client = self.create_client(GetRobotPose, '/kinematics/get_current_pose', callback_group=timer_cb_group)
        self.get_current_pose_client.wait_for_service()
        endpoint = self.send_request(self.get_current_pose_client, GetRobotPose.Request())
        pose_t = endpoint.pose.position
        pose_r = endpoint.pose.orientation
        

        self.endpoint = common.xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])
            
        # 获取标签数据(get tag data)
        t = time.time()
        self.tags = []
        self.calibration_step = 1
        while self.calibration_step == 1 and time.time() - t < 10:
            time.sleep(0.1)

        if len(self.tags) < 5:
            self.err_msg = "Time out, calibrate failed!!!"
            time.sleep(3)
            self.err_msg = None
            self.calibration_step = 0
            self.thread = None
            self.tags = []
            return

 
        # 识别区域中心位置标定(calibration of the center position in the recognition area)
        # 对多次识别的数据求均值(calculate the average of multiple recognition data)
        pose = map(lambda tag: common.xyz_rot_to_mat(tag.pose_t, tag.pose_R), self.tags) # 将所有位姿转为4x4齐次矩阵(convert all poses to 4x4 homogeneous matrices)
        vectors = map(lambda p: p.ravel(), pose) # 将矩阵展平为向量(flatten the matrix into a vector)
        avg_pose = np.mean(list(vectors), axis=0).reshape((4, 4))  # 求均值并重组为4x4矩阵(calculate the mean and reassemble into a 4x4 matrix)
        pose_end = np.matmul(self.hand2cam_tf_matrix, avg_pose)  # 转换到末端相对坐标(transform to end-effector relative coordinates)

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
        tvec_flattened = tvec.flatten().tolist()


        extristric = [
            tvec_flattened,  # 将 tvec 作为第一行
            rmat[0].tolist(),  # 第一行的旋转矩阵
            rmat[1].tolist(),  # 第二行的旋转矩阵
            rmat[2].tolist()   # 第三行的旋转矩阵
        ]
        self.extristric = np.array(extristric)
        # self.get_logger().info('extristric: ' + str(self.extristric))
    
        data = {
            'white_area_pose_cam': white_area_pose_cam,
            'white_area_pose_world': white_area_pose_world,
            'extristric': extristric,
        }
        self.update_yaml_data(data, self.config_path + self.config_file)
     
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
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)  
        else:
            data = {}  

        data.update(new_data)  

        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f)
    
        time.sleep(0.1)

    def start_calibration_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start calibration")
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
            else:
                msg = "Calibration..."
                self.get_logger().info(msg)
                response.success = False
                response.message = "stop"
                return response

    def enter_srv_callback(self,request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "loading calibration")
        # 获取和发布图像的topic(get and publish topic of image)
        self.image_sub = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/depth/camera_info', self.camera_info_callback, 1)
        set_servo_position(self.joints_pub, 2.0, ((1, 500), (2, 560), (3, 130), (4, 115), (5, 500), (10, 200)))  # 设置机械臂初始位置
        self.running = True
        self.thread_started = False
        response.success = True
        response.message = "start"
        return response

    def draw_retangle(self):

        white_area_center = self.white_area_pose_world.reshape(4, 4)
        white_area_cam = self.white_area_pose_cam.reshape(4, 4)
        euler_matrix = common.xyz_euler_to_mat((self.white_area_height / 2, self.white_area_width / 2 + 0.0, 0.0), (0, 0, 0))
        white_area_lt = np.matmul(white_area_center, common.xyz_euler_to_mat((self.white_area_height / 2, self.white_area_width / 2 + 0.0, 0.0), (0, 0, 0)))
        white_area_lb = np.matmul(white_area_center, common.xyz_euler_to_mat((-self.white_area_height / 2, self.white_area_width / 2 + 0.0, 0.0), (0, 0, 0)))
        white_area_rb = np.matmul(white_area_center, common.xyz_euler_to_mat((-self.white_area_height / 2, -self.white_area_width / 2 -0.0, 0.0), (0, 0, 0)))
        white_area_rt = np.matmul(white_area_center, common.xyz_euler_to_mat((self.white_area_height / 2, -self.white_area_width / 2 -0.0, 0.0), (0, 0, 0)))

        endpoint = self.send_request(self.get_current_pose_client, GetRobotPose.Request())
        pose_t = endpoint.pose.position
        pose_r = endpoint.pose.orientation

        self.endpoint = common.xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])
        corners_cam =  np.matmul(np.linalg.inv(np.matmul(self.endpoint, self.hand2cam_tf_matrix)), [white_area_lt, white_area_lb, white_area_rb, white_area_rt, white_area_center])
        corners_cam = np.matmul(np.linalg.inv(white_area_cam), corners_cam)
        corners_cam = corners_cam[:, :3, 3:].reshape((-1, 3))
        tvec = self.extristric[:1]  
        rmat = self.extristric[1:]  


        while self.K is None or self.D is None:
            time.sleep(0.5)

        center_imgpts, jac = cv2.projectPoints(corners_cam[-1:], np.array(rmat), np.array(tvec), self.K, self.D)
        self.center_imgpts = np.int32(center_imgpts).reshape(2)

        tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.030)

        imgpts, jac = cv2.projectPoints(corners_cam[:-1], np.array(rmat), np.array(tvec), self.K, self.D)

        self.imgpts = np.int32(imgpts).reshape(-1, 2)


    def camera_info_callback(self, msg):
        with self.lock:
            K = np.matrix(msg.k).reshape(1, -1, 3)
            D = np.array(msg.d)
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (640, 480), 0, (640, 480))
            self.K, self.D = np.matrix(new_K), np.zeros((5, 1))


    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "stop calibration")

        self.running = False
        try:
            if self.image_sub is not None:
                self.destroy_subscription(self.image_sub)
                self.destroy_subscription(self.camera_info_sub)
                self.image_sub = None
                self.camera_info_sub = None
        except Exception as e:
            self.get_logger().error(str(e))
        response.success = True
        response.message = "stop"
        return response  
    

    def image_callback(self, ros_image):
        # 将ros格式图像转换为opencv格式(convert the ros format image to opencv format)

        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if not self.thread_started:
            # 只在第一次调用时启动线程
            threading.Thread(target=self.image_processing, args=(ros_image,), daemon=True).start()
            self.thread_started = True 

        if self.image_queue.full():
            # # 如果队列已满，丢弃最旧的图像
            self.image_queue.get()
        # # 将图像放入队列
        self.image_queue.put(rgb_image)

        
    def image_processing(self, ros_image):
        while self.running:
            rgb_image = self.image_queue.get()
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
                        if len(self.tags) >= 10:
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
                    image_points, _ = cv2.projectPoints(world_points, self.extristric[1:].reshape(3, 3), self.extristric[:1], self.K, self.D)

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
                    cv2.drawContours(result_image, [self.imgpts], -1, (255, 255, 0), 2, cv2.LINE_AA) # 绘制矩形(draw rectangle)
                else:
                    msg = "Calibrating..."
                cv2.putText(result_image, msg, (5, result_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 6)
                cv2.putText(result_image, msg, (5, result_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
            cv2.drawContours(result_image, [self.imgpts], -1, (255, 255, 0), 2, cv2.LINE_AA) # 绘制矩形(draw rectangle)
            # 发布结果图像( publish the resulting image)
            self.result_image_pub.publish(self.bridge.cv2_to_imgmsg(result_image, "rgb8"))


def main():
    node = CalibrationNode('calibration')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()
