import cv2
import numpy as np
from matplotlib.pyplot import gray
from sklearn.linear_model import LinearRegression, RANSACRegressor

# 全局状态管理
current_state = type('', (), {})()
current_frame = type('', (), {'value': 0})

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
    numDisparities=128,
    blockSize=5,
    uniquenessRatio=15,
    speckleWindowSize=100,
    speckleRange=2,
    disp12MaxDiff=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    mode=cv2.STEREO_SGBM_MODE_HH4
)
R_cam_to_world = np.array([[9.95809060e-01, 5.65688399e-10, .14566442e-02],
                           [8.48472464e-02, 3.73247169e-01, 9.23843920e-01],
                           [3.41359309e-02, .27731990e-01, 3.71682912e-01]])
ground_normal = np.array([271.09875, 13.53952, 159.0065])


class CoordinateTransformer:
    def __init__(self):
        self.origin_cam = None  # 相机坐标系中的坐标 (mm)
        self.R_cam_to_world = np.array([
            [9.95809060e-01, 5.65688399e-10, .14566442e-02],
            [8.48472464e-02, 3.73247169e-01, 9.23843920e-01],
            [3.41359309e-02, .27731990e-01, 3.71682912e-01]
        ])  # 旋转矩阵
        self.ground_normal = np.array([271.09875, 13.53952, 159.0065])  # 地面法向量
        self.ground_normal = self.ground_normal / np.linalg.norm(self.ground_normal)  # 预先归一化
        self._cached_dirty = True  # 缓存状态标记

    def set_origin(self, point_cam):
        """设置原点坐标"""
        self.origin_cam = np.array(point_cam)
        self._cached_dirty = True  # 标记缓存需要更新

    def set_ground_normal(self, normal):
        """设置地面法向量并计算旋转矩阵"""
        normal_array = np.array(normal, dtype=np.float64)
        self.ground_normal = normal_array / np.linalg.norm(normal_array)
        self._compute_rotation()
        self._cached_dirty = True  # 标记缓存需要更新

    def _compute_rotation(self):
        # 与原来相同，但使用预先计算的归一化向量
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
        if self.origin_cam is None or self.R_cam_to_world is None:
            raise ValueError("原点或旋转矩阵未初始化")
        return np.dot(self.R_cam_to_world, (point_cam - self.origin_cam))


def draw_coordinate_axes(img, transformer):
    """在图像上绘制世界坐标系箭头 - 优化版本"""
    # 检查参数是否已初始化
    if transformer.origin_cam is None or transformer.R_cam_to_world is None:
        return

    # 定义三个轴端点（在相机坐标系中）
    axis_length = 100  # 毫米
    
    # 使用缓存避免重复计算
    if not hasattr(transformer, '_cached_axes_px') or transformer._cached_dirty:
        try:
            axes_cam = np.array([
                transformer.origin_cam,
                transformer.origin_cam + axis_length * transformer.R_cam_to_world.T[0],  # X轴
                transformer.origin_cam + axis_length * transformer.R_cam_to_world.T[1],  # Y轴
                transformer.origin_cam + axis_length * transformer.R_cam_to_world.T[2],  # Z轴
            ])
            
            # 将3D点投影到图像平面
            axes_px, _ = cv2.projectPoints(
                axes_cam,
                np.eye(3), np.zeros(3),
                left_camera_matrix, left_distortion
            )
            transformer._cached_axes_px = axes_px.astype(int)
            transformer._cached_dirty = False
        except Exception as e:
            print(f"坐标轴绘制错误: {str(e)}")
            return
    
    # 使用缓存的投影点绘制箭头
    axes_px = transformer._cached_axes_px
    origin = tuple(axes_px[0][0])
    cv2.arrowedLine(img, origin, tuple(axes_px[1][0]), (0, 0, 255), 2)  # X轴-红
    cv2.arrowedLine(img, origin, tuple(axes_px[2][0]), (0, 255, 0), 2)  # Y轴-绿
    cv2.arrowedLine(img, origin, tuple(axes_px[3][0]), (255, 0, 0), 2)  # Z轴-蓝


def onmouse_pick_points(event, x, y, flags, param):
    global transform_origin, axis_points, transformer
    
    # 只处理特定鼠标事件，减少不必要的计算
    if event != cv2.EVENT_LBUTTONDOWN:
        return
        
    imgL_rectified, imgR_rectified = param
    
    # 使用延迟计算 - 只有在确实需要时才计算视差
    # 只在指定范围内的点计算视差，避免边缘无效区域
    h, w = imgL_rectified.shape[:2]
    if x < 10 or y < 10 or x >= w-10 or y >= h-10:
        print("点击位置太靠近边缘")
        return
    
    # 计算该点周围小区域的视差，而不是整个图像
    roi_size = 100  # ROI大小
    x1, y1 = max(0, x-roi_size//2), max(0, y-roi_size//2)
    x2, y2 = min(w, x1+roi_size), min(h, y1+roi_size)
    
    roi_L = imgL_rectified[y1:y2, x1:x2]
    roi_R = imgR_rectified[y1:y2, x1:x2]
    
    # 计算ROI区域的视差
    disparity_roi = stereo.compute(roi_L, roi_R)
    disparity_float_roi = disparity_roi.astype(np.float32) / 16.0
    
    # 相对位置和点坐标计算
    local_x, local_y = x - x1, y - y1
    
    # 构建小的Q矩阵用于此ROI
    local_Q = Q.copy()  # 复制Q矩阵
    
    # 手动计算3D点坐标，避免计算整个图像
    point_z = disparity_float_roi[local_y, local_x]
    if point_z <= 0:  # 无效视差
        print("无效视差，无法计算3D坐标")
        return
        
    # 手动计算3D点坐标
    point_cam = np.zeros(3)
    point_cam[0] = ((x - local_Q[0,3]) * point_z) / local_Q[0,0]
    point_cam[1] = ((y - local_Q[1,3]) * point_z) / local_Q[1,1]
    point_cam[2] = point_z * local_Q[2,2] + local_Q[2,3]
    
    # 处理按键操作
    if flags & cv2.EVENT_FLAG_ALTKEY:
        # 深度过滤（仅采集1m-3m范围内的点）
        if 500 < np.linalg.norm(point_cam) < 3000:
            # 空间分布检查（与前5个点距离需大于50mm）
            if len(ground_points) < 5 or np.min(np.linalg.norm(np.array(ground_points) - point_cam, axis=1)) > 50:
                ground_points.append(point_cam)
                print('添加地面点', point_cam[0] / 1000, point_cam[1] / 1000, point_cam[2] / 1000)
    
    elif flags & cv2.EVENT_FLAG_SHIFTKEY:
        if not np.all(np.isfinite(point_cam)) or np.any(np.abs(point_cam) > 1e6):
            print("警告：无效坐标，请重新选择原点")
            return
        transformer.set_origin(point_cam)
        print(f"原点已设置: {point_cam / 1000} m")
    
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
    """PCA初筛 + RANSAC精修 - 优化版本"""
    points = np.array(points, dtype=np.float32)
    if len(points) < 3:
        raise ValueError("至少需要3个点来拟合平面")

    # 初步PCA拟合
    mean, eigvec = cv2.PCACompute(points, mean=None)
    normal_pca = eigvec[2]

    # RANSAC在PCA结果附近搜索 - 减少迭代次数提高效率
    inlier_threshold = 5.0  # 5mm误差阈值
    best_normal = normal_pca.copy()
    max_inliers = 0
    iteration_count = min(50, len(points) * 10)  # 动态调整迭代次数

    for _ in range(iteration_count):
        # 随机选择3个点生成平面假设
        indices = np.random.choice(len(points), 3, replace=False)
        sample = points[indices]
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-6:  # 处理退化情况
            continue
        n /= norm

        # 计算内点数量 - 矢量化计算
        dist = np.abs(np.sum((points - sample[0]) * n, axis=1))
        inliers = np.sum(dist < inlier_threshold)

        if inliers > max_inliers:
            max_inliers = inliers
            best_normal = n

    # 方向校正（假设相机Y轴向下）
    if best_normal[1] > 0:
        best_normal *= -1

    return best_normal


def detect_checkerboard(imgL_rectified, imgR_rectified):
    """棋盘格检测优化版本"""
    # 降采样提高速度
    scale = 0.5
    small_imgL = cv2.resize(imgL_rectified, None, fx=scale, fy=scale)
    
    # 先在低分辨率图像上检测
    ret_left, small_corners_left = cv2.findChessboardCorners(
        small_imgL, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
    )
    
    if not ret_left:
        return False, None, None
    
    # 在原始图像上精确定位
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_left = small_corners_left * (1/scale)
    corners_left = cv2.cornerSubPix(
        imgL_rectified, corners_left, (11, 11), (-1, -1), criteria
    )
    
    # 检测右视图角点
    small_imgR = cv2.resize(imgR_rectified, None, fx=scale, fy=scale)
    ret_right, small_corners_right = cv2.findChessboardCorners(
        small_imgR, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
    )
    
    if not ret_right:
        return False, corners_left, None
    
    # 右视图精确定位
    corners_right = small_corners_right * (1/scale)
    corners_right = cv2.cornerSubPix(
        imgR_rectified, corners_right, (11, 11), (-1, -1), criteria
    )
    
    return True, corners_left, corners_right


def visualize_ground_plane(img, transformer, camera_matrix, dist_coeffs, plane_size=2000, grid_step=500):
    """优化的地面可视化函数"""
    if transformer.origin_cam is None or transformer.R_cam_to_world is None:
        return img
    
    # 使用缓存避免重复计算
    if not hasattr(transformer, '_cached_grid_points') or transformer._cached_dirty:
        # 生成网格点（世界坐标系，Z=0）- 使用更少的点
        reduced_size = plane_size
        reduced_step = grid_step * 2  # 增大网格间距减少点数
        xx, yy = np.meshgrid(
            np.arange(-reduced_size // 2, reduced_size // 2 + 1, reduced_step),
            np.arange(-reduced_size // 2, reduced_size // 2 + 1, reduced_step)
        )
        zz = np.zeros_like(xx)
        points_world = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

        # 转换到相机坐标系
        R_world_to_cam = transformer.R_cam_to_world.T
        points_cam = (np.dot(R_world_to_cam, points_world.T)).T + transformer.origin_cam

        # 投影到图像平面
        points_img, _ = cv2.projectPoints(
            points_cam.astype(np.float32),
            np.zeros(3), np.zeros(3),
            camera_matrix, dist_coeffs
        )
        
        transformer._cached_grid_points = {
            'points': points_img.reshape(-1, 2).astype(int),
            'shape': xx.shape
        }
        transformer._cached_dirty = False
    
    # 使用缓存的网格点绘制
    points_img = transformer._cached_grid_points['points']
    n = transformer._cached_grid_points['shape'][0]
    
    # 绘制网格线 - 只绘制在图像内的部分
    color = (0, 255, 0)  # 绿色
    h, w = img.shape[:2]
    
    # 使用向量化操作检查点是否在图像内
    valid_points = (points_img[:, 0] >= 0) & (points_img[:, 0] < w) & \
                  (points_img[:, 1] >= 0) & (points_img[:, 1] < h)
    
    # 水平线和垂直线的绘制
    for i in range(n):
        for j in range(n - 1):
            idx1 = i * n + j
            idx2 = i * n + j + 1
            if idx1 < len(valid_points) and idx2 < len(valid_points) and \
               valid_points[idx1] and valid_points[idx2]:
                cv2.line(img, tuple(points_img[idx1]), tuple(points_img[idx2]), color, 1)
        
        for j in range(n):
            idx1 = j * n + i
            idx2 = (j + 1) * n + i
            if idx2 < len(valid_points) and idx1 < len(valid_points) and \
               valid_points[idx1] and valid_points[idx2]:
                cv2.line(img, tuple(points_img[idx1]), tuple(points_img[idx2]), color, 1)

    return img


def calculate_point_operation(xy, frame1, frame2):
    """优化的三维坐标处理函数"""
    global ground_points, transformer
    point_world = []
    
    # 预处理一次灰度转换
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    imgL_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    imgR_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)
    
    # 一次性计算视差图
    disparity = stereo.compute(imgL_rectified, imgR_rectified)
    disparity_float = disparity.astype(np.float32) / 16.0
    
    # 按需计算三维点
    for xy_temp in xy:
        if len(xy_temp) == 2:
            x, y = xy_temp
            
            # 边界检查
            if 0 <= x < imgL_rectified.shape[1] and 0 <= y < imgL_rectified.shape[0]:
                # 直接从已计算的视差图获取值
                disp_val = disparity_float[y, x]
                
                # 检查视差值有效性
                if disp_val > 0:
                    # 手动计算3D点，避免计算整个视差图的reprojectImageTo3D
                    point_cam = np.zeros(3)
                    point_cam[2] = disp_val * Q[2, 2] + Q[2, 3]  # z坐标 
                    point_cam[0] = ((x - Q[0, 3]) * point_cam[2]) / Q[0, 0]  # x坐标
                    point_cam[1] = ((y - Q[1, 3]) * point_cam[2]) / Q[1, 1]  # y坐标
                    
                    # 坐标转换
                    point_world.append((transformer.transform(point_cam) / 1000).tolist())
                else:
                    point_world.append([-1, -1, -1])  # 无效点
            else:
                point_world.append([-1, -1, -1])  # 超出边界的点
        else:
            point_world.append([-1, -1, -1])  # 格式不正确的点

    return point_world


# 全局初始化
ground_points = []
transformer = CoordinateTransformer()
transformer.set_ground_normal(ground_normal)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
transformer.set_origin(np.array([461.60626, 535.6015, 371.5321]))

if __name__ == '__main__':
    # 帧计数器
    frame_count = 0
    
    # 缓存
    cached_imgL_rectified = None
    cached_imgR_rectified = None
    
    # FPS计算相关变量
    fps = 0
    prev_time = cv2.getTickCount()
    fps_update_interval = 10  # 每10帧更新一次FPS
    
    while True:
        ret, frame = capture.read()
        if not ret:
            continue

        # 帧计数增加
        frame_count += 1
        
        # 分割左右视图
        frame1 = frame[:, :1280]
        frame2 = frame[:, 1280:]

        # 校正图像 - 每3帧完全处理一次
        process_full = frame_count % 3 == 0
        
        if process_full:
            # 完整处理
            imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            imgL_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
            imgR_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)
            
            # 更新缓存
            cached_imgL_rectified = imgL_rectified.copy()
            cached_imgR_rectified = imgR_rectified.copy()
        else:
            # 使用缓存数据
            if cached_imgL_rectified is not None and cached_imgR_rectified is not None:
                imgL_rectified = cached_imgL_rectified
                imgR_rectified = cached_imgR_rectified
            else:
                # 初始缓存为空，执行一次完整处理
                imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                imgL_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
                imgR_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)
                
                # 初始化缓存
                cached_imgL_rectified = imgL_rectified.copy()
                cached_imgR_rectified = imgR_rectified.copy()

        # 显示图像
        imgL_display = cv2.cvtColor(imgL_rectified, cv2.COLOR_GRAY2BGR)
        imgR_display = cv2.cvtColor(imgR_rectified, cv2.COLOR_GRAY2BGR)
        
        # 仅在完整处理帧上绘制坐标轴
        if process_full:
            draw_coordinate_axes(imgL_display, transformer)
        
        # 计算和显示FPS
        if frame_count % fps_update_interval == 0:
            curr_time = cv2.getTickCount()
            time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
            fps = fps_update_interval / time_diff
            print(f"FPS: {fps:.1f}")
            prev_time = curr_time
        
        # 在图像上显示FPS
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
                if len(ground_points) < 3:
                    print("需要至少3个点来拟合平面")
                    continue
                ground_normal = fit_ground_plane(ground_points)
                print(f"地面法向量: [{ground_normal[0]:.6f}, {ground_normal[1]:.6f}, {ground_normal[2]:.6f}]")
                transformer.set_ground_normal(ground_normal)
            except Exception as e:
                print(f"地面标定失败: {str(e)}")
        elif key == ord('x'):  # 按X键采集标定板方向
            # 强制完整处理当前帧
            imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            imgL_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
            imgR_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)
            
            # 检测棋盘格
            ret, corners_left, corners_right = detect_checkerboard(imgL_rectified, imgR_rectified)
            if not ret:
                print("未检测到完整棋盘格！")
                continue

            # 计算视差图
            disparity = stereo.compute(imgL_rectified, imgR_rectified)
            disparity_float = disparity.astype(np.float32) / 16.0
            
            # 只计算所需的两个角点的3D坐标，而不是整个图像
            pt1 = (int(corners_left[0][0][0]), int(corners_left[0][0][1]))
            pt2 = (int(corners_left[pattern_size[0] - 1][0][0]), int(corners_left[pattern_size[0] - 1][0][1]))
            
            # 手动计算3D点坐标
            point1 = np.zeros(3)
            point2 = np.zeros(3)
            
            # 检查点1在视差图内
            if 0 <= pt1[0] < disparity_float.shape[1] and 0 <= pt1[1] < disparity_float.shape[0]:
                disp1 = disparity_float[pt1[1], pt1[0]]
                if disp1 > 0:  # 视差有效
                    point1[2] = disp1 * Q[2, 2] + Q[2, 3]
                    point1[0] = ((pt1[0] - Q[0, 3]) * point1[2]) / Q[0, 0]
                    point1[1] = ((pt1[1] - Q[1, 3]) * point1[2]) / Q[1, 1]
            
            # 检查点2在视差图内
            if 0 <= pt2[0] < disparity_float.shape[1] and 0 <= pt2[1] < disparity_float.shape[0]:
                disp2 = disparity_float[pt2[1], pt2[0]]
                if disp2 > 0:  # 视差有效
                    point2[2] = disp2 * Q[2, 2] + Q[2, 3]
                    point2[0] = ((pt2[0] - Q[0, 3]) * point2[2]) / Q[0, 0]
                    point2[1] = ((pt2[1] - Q[1, 3]) * point2[2]) / Q[1, 1]
            
            ref_x_axis = point2 - point1
            norm = np.linalg.norm(ref_x_axis)
            
            if norm < 1e-6:
                print("无法计算有效的参考轴向量")
                continue
                
            ref_x_axis /= norm
            
            # 更新变换器
            transformer.ref_x_axis = ref_x_axis
            print(f"标定板X轴方向已设置: {ref_x_axis}")
            
        elif key == ord('r'):  # 清空地面点
            ground_points = []
            print("地面点已清空")
            
    # 释放资源
    capture.release()
    cv2.destroyAllWindows()