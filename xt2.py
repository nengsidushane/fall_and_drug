import cv2
import numpy as np
from matplotlib.pyplot import gray
from sklearn.linear_model import LinearRegression, RANSACRegressor

# 初始化相机参数和校正参数
left_camera_matrix = np.array([[650.8727267, 0, 632.0660297],
                               [0, 651.5738202, 358.6525714],
                               [0, 0, 1]])
right_camera_matrix = np.array([[653.5158803, 0, 642.6118299],
                                [0, 653.7522913, 353.6209308],
                                [0, 0, 1]])
left_distortion = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
right_distortion = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
R = np.array([[0.999921585, 0.001239856, -0.012461435],
              [-0.001141365, 0.999968082, 0.007907702],
              [0.012470841, -0.007892859, 0.999891085]])
T = np.array([[-61.59406398], [0.140562835], [-0.721770621]])
size = (1280, 720)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                            right_camera_matrix, right_distortion, size, R, T)
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
pattern_size = (11, 8)
blockSize = 5
img_channels = 3
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,  # 根据基线调整（Baseline=60mm时建议值）
    blockSize=5,
    uniquenessRatio=15,  # 提高以抑制误匹配
    speckleWindowSize=100,  # 降低椒盐噪声
    speckleRange=2,
    disp12MaxDiff=5,  # 左右视差一致性检查
    P1=8 * 3 * 5 ** 2,  # 平滑约束参数
    P2=32 * 3 * 5 ** 2,
    mode=cv2.STEREO_SGBM_MODE_HH4
)
R_cam_to_world = np.array([[9.95809060e-01, 5.65688399e-10, .14566442e-02],
                           [8.48472464e-02, 3.73247169e-01, 9.23843920e-01],
                           [3.41359309e-02, .27731990e-01, 3.71682912e-01]])
ground_normal = np.array([271.09875, 13.53952, 159.0065])

# 添加到全局变量区域
# 缓存预计算结果
CACHED_AXES_CAM = None  # 缓存坐标轴相机坐标
CACHED_AXES_PX = None   # 缓存坐标轴像素坐标
prev_origin = None      # 记录上次坐标原点
prev_R = None           # 记录上次旋转矩阵
processing_scale = 0.5   # 处理时的图像缩放比例
display_scale = 1.0      # 显示时的图像缩放比例
class CoordinateTransformer:
    def __init__(self):
        self.origin_cam = None  # 相机坐标系中的坐标 (mm)
        self.R_cam_to_world = np.array([[9.95809060e-01, 5.65688399e-10, .14566442e-02],
                               [8.48472464e-02, 3.73247169e-01, 9.23843920e-01],
                               [3.41359309e-02, .27731990e-01, 3.71682912e-01]])  # 旋转矩阵
        self.ground_normal = np.array([271.09875, 13.53952, 159.0065])  # 地面法向量 (单位向量)
        # 预计算变换矩阵的逆矩阵
        self.R_world_to_cam = self.R_cam_to_world.T

    def set_origin(self, point_cam):
        """设置原点坐标"""
        self.origin_cam = np.array(point_cam)

    def set_ground_normal(self, normal):
        """设置地面法向量并计算旋转矩阵"""
        self.ground_normal = normal / np.linalg.norm(normal)
        self._compute_rotation()

    def _compute_rotation(self):
        z_axis = self.ground_normal

        # 使用标定板自身X方向作为参考
        if hasattr(self, 'ref_x_axis'):
            x_axis = self.ref_x_axis - np.dot(self.ref_x_axis, z_axis) * z_axis
        else:
            x_axis = np.array([1, 0, 0]) - np.dot([1, 0, 0], z_axis) * z_axis

        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # 正交化修正
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        self.R_cam_to_world = np.vstack([x_axis, y_axis, z_axis]).T

    def transform(self, point_cam):
        """将相机坐标系下的点转换到世界坐标系"""
        if self.origin_cam is None:
            raise ValueError("原点未初始化")
        # 使用矩阵乘法而不是@操作符，并避免不必要的副本创建
        return np.matmul(self.R_cam_to_world, (point_cam - self.origin_cam))


# 优化绘制坐标轴函数
def draw_coordinate_axes(img, transformer):
    """优化的坐标轴绘制函数，使用缓存避免重复计算"""
    global CACHED_AXES_CAM, CACHED_AXES_PX, prev_origin, prev_R
    
    # 检查参数是否已初始化
    if transformer.origin_cam is None:
        return

    # 检查是否需要重新计算投影点
    recalculate = (CACHED_AXES_PX is None or 
                  prev_origin is None or 
                  prev_R is None or
                  not np.array_equal(transformer.origin_cam, prev_origin) or
                  not np.array_equal(transformer.R_cam_to_world, prev_R))
                  
    try:
        if recalculate:
            # 预计算轴端点，避免重复计算
            axis_length = 100  # 毫米
            origin_cam = transformer.origin_cam
            x_axis = origin_cam + axis_length * transformer.R_cam_to_world.T[0]
            y_axis = origin_cam + axis_length * transformer.R_cam_to_world.T[1]
            z_axis = origin_cam + axis_length * transformer.R_cam_to_world.T[2]
            
            # 直接构建点阵列，避免额外的numpy操作
            CACHED_AXES_CAM = np.array([origin_cam, x_axis, y_axis, z_axis])
            
            # 将3D点投影到图像平面
            CACHED_AXES_PX, _ = cv2.projectPoints(
                CACHED_AXES_CAM,
                np.eye(3), np.zeros(3),
                left_camera_matrix, left_distortion
            )
            CACHED_AXES_PX = CACHED_AXES_PX.astype(int)
            
            # 更新缓存状态
            prev_origin = transformer.origin_cam.copy()
            prev_R = transformer.R_cam_to_world.copy()
            
        # 绘制箭头
        origin = tuple(CACHED_AXES_PX[0][0])
        cv2.arrowedLine(img, origin, tuple(CACHED_AXES_PX[1][0]), (0, 0, 255), 2)  # X轴-红
        cv2.arrowedLine(img, origin, tuple(CACHED_AXES_PX[2][0]), (0, 255, 0), 2)  # Y轴-绿
        cv2.arrowedLine(img, origin, tuple(CACHED_AXES_PX[3][0]), (255, 0, 0), 2)  # Z轴-蓝
    except Exception as e:
        print(f"坐标轴绘制错误: {str(e)}")

# 优化鼠标回调函数，按需计算视差图
def onmouse_pick_points(event, x, y, flags, param):
    global transform_origin, axis_points, transformer, disparity_map, threeD_map, processing_scale
    
    if event == cv2.EVENT_LBUTTONDOWN:
        imgL_rectified, imgR_rectified = param
        
        # 坐标缩放处理（如果使用了降采样）
        scaled_x = int(x / processing_scale)
        scaled_y = int(y / processing_scale)
        
        # 检查坐标是否在有效范围内
        h, w = imgL_rectified.shape[:2]
        if not (0 <= scaled_x < w and 0 <= scaled_y < h):
            print("点击位置超出有效范围")
            return

        # 视差图和3D地图按需计算，避免重复计算
        if disparity_map is None or threeD_map is None:
            # 使用更快的视差计算参数进行计算
            disparity_map = stereo.compute(imgL_rectified, imgR_rectified)
            disparity_float = disparity_map.astype(np.float32) / 16.0
            threeD_map = cv2.reprojectImageTo3D(disparity_float, Q, handleMissingValues=True)

        # 获取当前点的坐标（单位：毫米）
        point_cam = np.array([
            threeD_map[scaled_y][scaled_x][0],
            threeD_map[scaled_y][scaled_x][1],
            threeD_map[scaled_y][scaled_x][2]
        ])
        
        # 检查点是否有效（避免无效点计算）
        if not np.all(np.isfinite(point_cam)) or np.any(np.abs(point_cam) > 5000):
            print("选择的点无效，请重新选择")
            return
            
        # 采集地面点（按住Alt键时触发）
        if flags & cv2.EVENT_FLAG_ALTKEY:
            # 深度过滤（仅采集1m-3m范围内的点）
            point_norm = np.linalg.norm(point_cam)
            if 500 < point_norm < 3000:
                # 空间分布检查（与前5个点距离需大于50mm）
                if len(ground_points) < 5 or np.min(np.linalg.norm(ground_points - point_cam, axis=1)) > 50:
                    ground_points.append(point_cam)
                    print('添加地面点', point_cam[0] / 1000, point_cam[1] / 1000, point_cam[2] / 1000)
        # 按下'Shift'键时设置原点
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            transformer.set_origin(point_cam)
            print(f"原点已设置: {point_cam / 1000} m")
        # 按下'Ctrl'键时验证地面坐标系Z轴
        elif flags & cv2.EVENT_FLAG_CTRLKEY:
            try:
                point_world = transformer.transform(point_cam)
                print(f"转换后坐标 (X,Y,Z): {point_world / 1000} m")
            except Exception as e:
                print(e)


def fit_ground_plane_opencv(points):
    """使用OpenCV的PCA方法直接拟合地面平面"""
    points = np.array(points, dtype=np.float32)
    if len(points) < 3:
        raise ValueError("至少需要3个点来拟合平面")

    # PCA分析地面点云
    mean, eigenvectors = cv2.PCACompute(points, mean=None)

    # 法向量为最小特征值对应的特征向量
    normal = eigenvectors[2]
    normal /= np.linalg.norm(normal)

    # 确保法向量方向朝上（根据场景调整符号）
    if normal[1] > 0:
        normal *= -1

    return normal


def fit_ground_plane(points):
    """PCA初筛 + RANSAC精修"""
    points = np.array(points, dtype=np.float32)

    # 初步PCA拟合
    mean, eigvec = cv2.PCACompute(points, mean=None)
    normal_pca = eigvec[2]

    # RANSAC在PCA结果附近搜索
    inlier_threshold = 5.0  # 5mm误差阈值
    best_normal = normal_pca.copy()
    max_inliers = 0

    for _ in range(100):
        # 随机选择3个点生成平面假设
        sample = points[np.random.choice(len(points), 3, replace=False)]
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        n = np.cross(v1, v2)
        n /= np.linalg.norm(n)

        # 计算内点数量
        dist = np.abs((points - sample[0]) @ n)
        inliers = np.sum(dist < inlier_threshold)

        if inliers > max_inliers:
            max_inliers = inliers
            best_normal = n

    # 方向校正（假设相机Y轴向下）
    if best_normal[1] > 0:
        best_normal *= -1

    return best_normal


def detect_checkerboard(imgL_rectified, imgR_rectified):
    # 检测左视图角点
    ret_left, corners_left = cv2.findChessboardCorners(
        imgL_rectified, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
    )

    # 检测右视图角点（需立体校正正确）
    if ret_left:
        ret_right, corners_right = cv2.findChessboardCorners(
            imgR_rectified, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        )
        return ret_left and ret_right, corners_left, corners_right
    return False, None, None


def visualize_ground_plane(img, transformer, camera_matrix, dist_coeffs, plane_size=2000, grid_step=500):
    """
    在图像上绘制地面网格
    :param img: 原始图像（BGR格式）
    :param transformer: CoordinateTransformer实例
    :param camera_matrix: 相机内参矩阵
    :param dist_coeffs: 畸变系数
    :param plane_size: 地面网格尺寸（毫米），默认2m x 2m
    :param grid_step: 网格线间隔（毫米），默认500mm
    """
    if transformer.origin_cam is None or transformer.R_cam_to_world is None:
        return img

    # 生成网格点（世界坐标系，Z=0）
    xx, yy = np.meshgrid(
        np.arange(-plane_size // 2, plane_size // 2 + 1, grid_step),
        np.arange(-plane_size // 2, plane_size // 2 + 1, grid_step)
    )
    zz = np.zeros_like(xx)
    points_world = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # 转换到相机坐标系
    R_world_to_cam = transformer.R_cam_to_world.T
    points_cam = (R_world_to_cam @ points_world.T).T + transformer.origin_cam

    # 投影到图像平面
    points_img, _ = cv2.projectPoints(
        points_cam.astype(np.float32),
        np.zeros(3), np.zeros(3),
        camera_matrix, dist_coeffs
    )
    points_img = points_img.reshape(-1, 2).astype(int)

    # 绘制网格线
    color = (0, 255, 0)  # 绿色
    n = xx.shape[0]
    for i in range(n):
        for j in range(n - 1):
            idx1 = i * n + j
            idx2 = i * n + j + 1
            if 0 <= points_img[idx1][0] < img.shape[1] and 0 <= points_img[idx1][1] < img.shape[0] and \
                    0 <= points_img[idx2][0] < img.shape[1] and 0 <= points_img[idx2][1] < img.shape[0]:
                cv2.line(img, tuple(points_img[idx1]), tuple(points_img[idx2]), color, 1)

        for j in range(n):
            idx1 = j * n + i
            idx2 = (j + 1) * n + i
            if idx2 < len(points_img) and \
                    0 <= points_img[idx1][0] < img.shape[1] and 0 <= points_img[idx1][1] < img.shape[0] and \
                    0 <= points_img[idx2][0] < img.shape[1] and 0 <= points_img[idx2][1] < img.shape[0]:
                cv2.line(img, tuple(points_img[idx1]), tuple(points_img[idx2]), color, 1)

    return img
def resize_frame(frame, scale_factor=0.5):
    """缩小图像以提高处理速度"""
    if scale_factor == 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, 
                     interpolation=cv2.INTER_AREA)

def calculate_point_operation(xy, frame1, frame2):
    """
    优化的三维坐标处理函数
    :param xy: 二维坐标数组
    :param frame1: 左视图原始图像
    :param frame2: 右视图原始图像
    """
    global ground_points, transformer
    
    # 如果坐标为空，快速返回
    if not xy:
        return []
        
    # 一次性处理所有图像转换，避免重复操作
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    imgL_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_NEAREST)  # 更快的插值
    imgR_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_NEAREST)
    
    # 一次性计算视差图，多点共用
    disparity = stereo_fast.compute(imgL_rectified, imgR_rectified)  # 使用更快的立体匹配
    disparity_float = disparity.astype(np.float32) / 16.0
    threeD = cv2.reprojectImageTo3D(disparity_float, Q, handleMissingValues=True)
    
    point_world = []
    for xy_temp in xy:
        if len(xy_temp) == 2:
            x, y = xy_temp
            
            # 检查坐标是否在有效范围内
            if 0 <= y < threeD.shape[0] and 0 <= x < threeD.shape[1]:
                # 获取当前点坐标（单位：毫米）
                point_cam = threeD[y][x]
                
                # 检查点是否有效
                if np.all(np.isfinite(point_cam)) and np.all(np.abs(point_cam) < 5000):
                    try:
                        # 坐标转换验证
                        point_world.append((transformer.transform(point_cam) / 1000).tolist())
                    except:
                        point_world.append([-1, -1, -1])
                else:
                    point_world.append([-1, -1, -1])
            else:
                point_world.append([-1, -1, -1])
        else:
            point_world.append([-1, -1, -1])

    return point_world
def create_optimized_stereo_matchers():
    """创建优化的立体匹配器"""
    # 高质量但较慢的SGBM
    stereo_hq = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        uniquenessRatio=10,  # 降低以提高速度
        speckleWindowSize=50,  # 降低以提高速度
        speckleRange=2,
        disp12MaxDiff=5,  # 左右视差一致性检查
        P1=8 * 3 * 5 ** 2,  
        P2=32 * 3 * 5 ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM  # 使用更快的SGBM模式
    )
    
    # 速度快但质量较低的BM
    stereo_fast = cv2.StereoBM_create(
        numDisparities=128,
        blockSize=15  # 使用较大的块大小以平衡质量和速度
    )
    
    return stereo_hq, stereo_fast

# 全局初始化
ground_points = []
transformer = CoordinateTransformer()
# 地面法向量
# ground_normal = np.array([0, 0, 1])
transformer.set_ground_normal(ground_normal)

transformer.set_origin(np.array([461.60626, 535.6015, 371.5321]))
# 优化主循环
if __name__ == '__main__':
    # FPS calculation variables
    fps_counter = 0
    fps = 0
    fps_start_time = cv2.getTickCount()
    
    # 立体匹配配置 - 使用更快的BM算法作为备选
    fast_stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)
    
    # 控制变量
    use_fast_stereo = True  # 使用快速立体匹配
    skip_frames = 0         # 每n帧处理1帧
    frame_count = 0
    last_print_time = 0     # 限制打印频率
    
    # 提前分配重复使用的缓冲区
    imgL_buffer = None
    imgR_buffer = None
    
    while True:
        frame_count += 1
        ret, frame = capture.read()
        if not ret:
            continue

        # 跳帧处理
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            # 最小处理：只显示未校正的左视图和FPS
            disp_frame = frame[:, :1280].copy()
            cv2.putText(disp_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("left", disp_frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue
            
        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 5:  # Update FPS every 5 frames for stability
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - fps_start_time) / cv2.getTickFrequency()
            fps = fps_counter / elapsed_time
            fps_counter = 0
            fps_start_time = current_time
            
        # 打印FPS的频率限制（每秒最多打印一次）
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if current_time - last_print_time > 1.0:  # 每秒最多打印一次
            print("FPS:", fps)
            last_print_time = current_time

        # 分割左右视图
        frame1 = frame[:, :1280]
        frame2 = frame[:, 1280:]

        # 可选：降低校正处理的分辨率
        if processing_scale != 1.0:
            frame1 = resize_frame(frame1, processing_scale)
            frame2 = resize_frame(frame2, processing_scale)
            
        # 校正图像 - 使用更快的插值方法
        if imgL_buffer is None or imgL_buffer.shape[:2] != frame1.shape[:2]:
            imgL_buffer = np.empty_like(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY))
            imgR_buffer = np.empty_like(imgL_buffer)
            
        # 转换为灰度图像
        imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY, dst=imgL_buffer)
        imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY, dst=imgR_buffer)
        
        # 使用更快的插值方法进行校正
        imgL_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_NEAREST)
        imgR_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_NEAREST)

        # 重置视差图缓存（仅在点击操作时计算）
        disparity_map = None
        threeD_map = None

        # 使用灰度图像显示，避免不必要的颜色转换
        # 创建3通道显示图像，避免每次从灰度转彩色
        imgL_display = np.empty((imgL_rectified.shape[0], imgL_rectified.shape[1], 3), dtype=np.uint8)
        imgL_display[:,:,0] = imgL_display[:,:,1] = imgL_display[:,:,2] = imgL_rectified
        
        # 只在必要时绘制坐标轴
        if transformer.origin_cam is not None:
            draw_coordinate_axes(imgL_display, transformer)
            
        # 添加FPS显示
        cv2.putText(imgL_display, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示图像
        cv2.imshow("left", imgL_display)
        cv2.setMouseCallback("left", onmouse_pick_points, (imgL_rectified, imgR_rectified))
        
        # 处理键盘输入
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('g'):  # 开始采集地面点
            collecting_ground_points = True
            ground_points = []
            print("开始采集地面点：按住Alt+左键点击地面区域")
        elif key == ord('f'):  # 拟合平面
            try:
                ground_normal = fit_ground_plane(ground_points)
                print(f"地面法向量: [{ground_normal[0]:.6f}, {ground_normal[1]:.6f}, {ground_normal[2]:.6f}]")
                transformer.set_ground_normal(ground_normal)
            except Exception as e:
                print(f"地面标定失败: {str(e)}")
        elif key == ord('s'):  # 切换立体匹配算法（快速/精确）
            use_fast_stereo = not use_fast_stereo
            print(f"使用{'快速' if use_fast_stereo else '精确'}立体匹配")
        elif key == ord('r'):  # 调整分辨率
            processing_scale = 1.0 if processing_scale < 1.0 else 0.5
            print(f"处理分辨率: {processing_scale}")
            # 清除缓存，以便在新分辨率下重新分配
            imgL_buffer = None
            imgR_buffer = None
            CACHED_AXES_PX = None
        elif key == ord('k'):  # 调整跳帧数
            skip_frames = (skip_frames + 1) % 4  # 0,1,2,3循环
            print(f"跳帧设置: 每{skip_frames+1}帧处理1帧")