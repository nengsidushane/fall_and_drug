import cv2
import numpy as np
from matplotlib.pyplot import gray
from sklearn.linear_model import LinearRegression, RANSACRegressor
import threading
import queue
import time # For sleep
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
# blockSize = 5 # BlockSize 保持较小可能更好
img_channels = 3
# --- 更激进的 SGBM 参数 ---
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # 大幅减少视差范围 (原为 128)
    blockSize=5,        # 保持较小值
    uniquenessRatio=10, # 稍微降低 (原为 15)
    speckleWindowSize=50,# 减少 (原为 100)
    speckleRange=2,     # 保持
    disp12MaxDiff=10,   # 稍微放宽 (原为 5)
    P1=4 * 3 * 5**2,    # 减少 P1 P2 (原为 8*..., 32*...)
    P2=16 * 3 * 5**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY # 尝试更快的模式 (原为 MODE_HH4)
    # mode=cv2.STEREO_SGBM_MODE_HH # 另一个可选的快速模式
)
R_cam_to_world = np.array([[9.95809060e-01, 5.65688399e-10, .14566442e-02],
                           [8.48472464e-02, 3.73247169e-01, 9.23843920e-01],
                           [3.41359309e-02, .27731990e-01, 3.71682912e-01]])
ground_normal = np.array([271.09875, 13.53952, 159.0065])


class CoordinateTransformer:
    def __init__(self):
        self.origin_cam = None  # 相机坐标系中的坐标 (mm)
        self.R_cam_to_world = [[9.95809060e-01, 5.65688399e-10, .14566442e-02],
                               [8.48472464e-02, 3.73247169e-01, 9.23843920e-01],
                               [3.41359309e-02, .27731990e-01, 3.71682912e-01]]  # 旋转矩阵
        self.ground_normal = [271.09875, 13.53952, 159.0065]  # 地面法向量 (单位向量)

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
        # print("origin_cam:",self.origin_cam)
        # print("R_cam_to_world:",self.R_cam_to_world)
        if self.origin_cam is None or self.R_cam_to_world is None:
            raise ValueError("原点或旋转矩阵未初始化")
        return self.R_cam_to_world @ (point_cam - self.origin_cam)


def draw_coordinate_axes(img, transformer):
    """在图像上绘制世界坐标系箭头"""
    # 检查参数是否已初始化
    if transformer.origin_cam is None or transformer.R_cam_to_world is None:
        # 显示提示信息
        # print('未初始化原点或旋转矩阵！按Shift+左键设置原点，按G采集地面点，按F拟合法向量')
        # cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return

    # 定义三个轴端点（在相机坐标系中）
    axis_length = 100  # 毫米
    try:
        axes_cam = np.array([
            transformer.origin_cam,
            transformer.origin_cam + axis_length * transformer.R_cam_to_world.T[0],  # X轴
            transformer.origin_cam + axis_length * transformer.R_cam_to_world.T[1],  # Y轴
            transformer.origin_cam + axis_length * transformer.R_cam_to_world.T[2],  # Z轴
        ])
    except Exception as e:
        print(f"坐标轴绘制错误: {str(e)}")
        return

    # 将3D点投影到图像平面
    axes_px, _ = cv2.projectPoints(
        axes_cam,
        np.eye(3), np.zeros(3),
        left_camera_matrix, left_distortion
    )
    axes_px = axes_px.astype(int)

    # 绘制箭头
    origin = tuple(axes_px[0][0])
    cv2.arrowedLine(img, origin, tuple(axes_px[1][0]), (0, 0, 255), 2)  # X轴-红
    cv2.arrowedLine(img, origin, tuple(axes_px[2][0]), (0, 255, 0), 2)  # Y轴-绿
    cv2.arrowedLine(img, origin, tuple(axes_px[3][0]), (255, 0, 0), 2)  # Z轴-蓝


def onmouse_pick_points(event, x, y, flags, param):
    """
    鼠标回调函数 - Optimized
    :param param: 字典，应包含预计算的 'threeD' 点云图, e.g., {"threeD": threeD_map}
    """
    global ground_points, transformer # 确保 ground_points 在全局可访问
    if event == cv2.EVENT_LBUTTONDOWN:
        if "threeD" not in param:
            print("错误：未在参数中找到'threeD'点云图")
            return

        threeD = param["threeD"] # 获取预计算的三维点云图

        # 检查点击坐标是否在图像范围内
        if 0 <= y < threeD.shape[0] and 0 <= x < threeD.shape[1]:
            # 直接从三维点云图中获取坐标 (单位：毫米)
            point_cam = threeD[y, x]

            # 检查获取的坐标是否有效 (非inf, 非nan, 非过大值)
            if not np.all(np.isfinite(point_cam)) or np.any(np.abs(point_cam) > 1e6):
                print("警告：选定点的深度值无效或超出范围，请重新选择。")
                return

            # --- 后续逻辑与原函数相同 ---

            # 采集地面点（按住Alt键时触发）
            if flags & cv2.EVENT_FLAG_ALTKEY:
                # 深度过滤（仅采集500mm-3000mm范围内的点）
                depth = np.linalg.norm(point_cam)
                if 500 < depth < 3000:
                    # 空间分布检查（与前5个点距离需大于50mm）
                    is_far_enough = True
                    if len(ground_points) >= 5:
                         # 计算与最近5个点的最小距离
                         min_dist = np.min(np.linalg.norm(np.array(ground_points[-5:]) - point_cam, axis=1))
                         if min_dist <= 50:
                             is_far_enough = False

                    if is_far_enough:
                        ground_points.append(point_cam)
                        print(f'添加地面点 {len(ground_points)}: {point_cam / 1000} m')
                    else:
                        print("警告：新点离最近的5个点太近 (<50mm)")
                else:
                    print(f"警告：点深度 {depth:.1f} mm 不在有效范围 (500-3000mm)")


            # 按下'Shift'键时设置原点
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                # 之前的检查已确保 point_cam 是有效的
                transformer.set_origin(point_cam)
                print(f"原点已设置: {point_cam / 1000} m")

            # 按下'Ctrl'键时验证地面坐标系Z轴
            elif flags & cv2.EVENT_FLAG_CTRLKEY:
                try:
                    point_world = transformer.transform(point_cam)
                    print(f"相机坐标: {point_cam / 1000} m -> 世界坐标: {point_world / 1000} m")
                except ValueError as e: # 捕获 transform 可能抛出的 ValueError
                    print(f"坐标转换错误: {e}")
                except Exception as e: # 捕获其他潜在错误
                    print(f"处理点击时发生未知错误: {e}")
            # else: # 普通左键点击，可以添加默认行为或提示
            #     print(f"点击坐标 (像素): ({x}, {y}), 相机坐标: {point_cam / 1000} m")

        else:
            print(f"警告：点击坐标 ({x}, {y}) 超出图像范围 ({threeD.shape[1]}x{threeD.shape[0]})")



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


def calculate_point_operation(xy, threeD, transformer):
    """
    三维坐标处理函数 - Optimized
    :param xy: list of [x, y] pixel coordinates.
    :param threeD: Precomputed 3D map (output of cv2.reprojectImageTo3D).
    :param transformer: Initialized CoordinateTransformer instance.
    :return: list of corresponding world coordinates [X, Y, Z] in meters,
             or specific negative values indicating errors.
             [-1,-1,-1]: Invalid depth at the pixel.
             [-2,-2,-2]: Coordinate transformation error (e.g., origin not set).
             [-3,-3,-3]: Other unexpected error during processing.
             [-4,-4,-4]: Pixel coordinates out of image bounds.
             [-5,-5,-5]: Invalid input format in xy list.
    """
    point_world = []
    if threeD is None or transformer is None:
        print("错误: calculate_point_operation 缺少 threeD 或 transformer 参数。")
        # 根据输入xy的长度返回错误标记列表
        return [[-3, -3, -3]] * len(xy)

    height, width = threeD.shape[:2]

    for xy_temp in xy:
        if isinstance(xy_temp, (list, tuple)) and len(xy_temp) == 2:
            try:
                x, y = map(int, xy_temp) # Ensure integer coordinates
            except (ValueError, TypeError):
                 point_world.append([-5, -5, -5]) # Invalid format within list
                 continue

            # Check bounds
            if 0 <= y < height and 0 <= x < width:
                point_cam = threeD[y, x]

                # Check for invalid points (inf, nan, large values)
                if np.all(np.isfinite(point_cam)) and not np.any(np.abs(point_cam) > 1e6):
                    try:
                        # Coordinate transformation
                        transformed_point = transformer.transform(point_cam) / 1000.0 # Convert mm to m
                        point_world.append(transformed_point.tolist())
                    except ValueError as e:
                        # print(f"坐标转换错误 for pixel ({x},{y}): {e}")
                        point_world.append([-2, -2, -2]) # Indicate transformation error
                    except Exception as e:
                        # print(f"处理点 ({x},{y}) 时发生未知错误: {e}")
                        point_world.append([-3, -3, -3]) # Indicate other error
                else:
                    # Invalid depth point
                    point_world.append([-1, -1, -1])
            else:
                # Point out of bounds
                 point_world.append([-4, -4, -4])
        else:
            # Invalid input format for this item in xy list
            point_world.append([-5, -5, -5])

    return point_world
class FrameReader:
    """
    一个使用独立线程从摄像头读取帧的类，以避免阻塞主处理循环。
    使用 LIFO 队列确保处理线程总是获取最新的可用帧。
    """
    def __init__(self, src=0, width=2560, height=720, max_queue_size=2):
        """
        初始化 FrameReader。
        :param src: 摄像头源索引或视频文件路径。
        :param width: 请求的帧宽度。
        :param height: 请求的帧高度。
        :param max_queue_size: 存储帧的队列最大尺寸。
        """
        self.src = src
        self.width = width
        self.height = height
        self.capture = cv2.VideoCapture(self.src)
        if not self.capture.isOpened():
             raise IOError(f"无法打开摄像头 {self.src}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # 使用后进先出队列 (LifoQueue) 来优先处理最新帧
        self.frame_queue = queue.LifoQueue(maxsize=max_queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # 设置为守护线程，主程序退出时自动结束

    def start(self):
        """启动读取线程。"""
        self.stopped = False
        self.thread.start()
        print("帧读取线程已启动。")
        return self

    def update(self):
        """线程运行的函数，持续读取帧并放入队列。"""
        while not self.stopped:
            ret, frame = self.capture.read()
            if not ret:
                print("帧读取线程：无法抓取帧，停止线程。")
                self.stop() # 如果读取失败则停止线程
                break
            # 如果队列满了，先取出一个旧帧，再放入新帧
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait() # 丢弃最旧的帧
                except queue.Empty:
                    pass # 理论上不应发生，但以防万一
            try:
                self.frame_queue.put(frame, block=False) # 非阻塞放入新帧
            except queue.Full:
                pass # 如果放入时队列又满了（不太可能），则放弃此帧

            # 可以取消注释下面一行，如果读取线程CPU占用过高
            # time.sleep(0.001)

    def read(self):
        """
        从队列中获取最新的帧。
        :return: (bool, frame) 元组，表示是否成功获取帧以及帧本身。
                 如果队列为空或超时，返回 (False, None)。
        """
        try:
            # 等待最多1秒获取帧，避免无限阻塞
            return True, self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            # print("帧读取线程：队列为空，可能读取线程已停止或处理过快。")
            return False, None # 指示失败或超时

    def stop(self):
        """停止读取线程并释放摄像头资源。"""
        print("正在停止帧读取线程...")
        self.stopped = True
        # 等待线程结束（设置短暂超时）
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.capture.isOpened():
            self.capture.release()
        print("帧读取线程已停止，摄像头已释放。")

# --- 全局初始化 (部分移至主块内) ---
ground_points = []
transformer = CoordinateTransformer()
# 地面法向量可以在启动后设置或使用默认值
transformer.set_ground_normal(ground_normal)
if __name__ == '__main__':
    # --- 启动帧读取线程 ---
    try:
        frame_reader = FrameReader(src=0, width=2560, height=720, max_queue_size=1).start() # 减小队列大小
        print("等待摄像头初始化...")
        time.sleep(1.5)
        transformer.set_origin(np.array([461.60626, 535.6015, 371.5321]))
        print("摄像头初始化完成。")
    except IOError as e:
        print(f"错误：无法初始化摄像头: {e}")
        exit()
    except Exception as e:
        print(f"错误：启动 FrameReader 时发生未知错误: {e}")
        exit()

    # FPS 计算变量
    processed_frame_count = 0
    fps = 0
    start_time = cv2.getTickCount()

    # --- 优化: 帧降采样 (跳帧处理) ---
    frame_skip = 1 # 尝试不跳帧，看是否是跳帧逻辑导致变慢
    fetch_count = 0

    # --- 优化: 图像降采样因子 ---
    downscale_factor = 2 # 降采样为原尺寸的 1/2 (长宽各一半)
    if downscale_factor > 1:
        # 调整 Q 矩阵以匹配降采样后的尺寸
        Q_downscaled = Q.copy()
        Q_downscaled[0, 3] /= downscale_factor
        Q_downscaled[1, 3] /= downscale_factor
        Q_downscaled[2, 3] /= downscale_factor
        Q_downscaled[3, 3] /= downscale_factor
        # Q[3, 2] (包含 1/Tx) 不变
        print("使用降采样，已调整Q矩阵。")
    else:
        Q_downscaled = Q # 不降采样则使用原始Q矩阵

    # 预分配变量
    imgL_rectified = None
    imgR_rectified = None
    threeD = None
    imgL_display = None

    print("开始主循环...")
    while True:
        ret, frame = frame_reader.read()

        if not ret:
            if not frame_reader.thread.is_alive() and not frame_reader.stopped:
                print("错误：帧读取线程意外停止！正在退出...")
                break
            key = cv2.waitKey(1)
            if key == ord('q'):
                 break
            continue

        fetch_count += 1
        if fetch_count % frame_skip != 0:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        processed_frame_count += 1

        # --- 计算和显示 FPS ---
        current_tick = cv2.getTickCount()
        elapsed_seconds_total = (current_tick - start_time) / cv2.getTickFrequency()
        if elapsed_seconds_total >= 1.0:
            fps = processed_frame_count / elapsed_seconds_total
            print(f"Processed FPS: {fps:.2f}")
            start_time = current_tick
            processed_frame_count = 0

        # --- 计算密集型步骤 ---
        try:
            frame1 = frame[:, :1280]
            frame2 = frame[:, 1280:]
            imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # --- 使用更快的插值 ---
            imgL_rectified_full = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_NEAREST)
            imgR_rectified_full = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_NEAREST)

            # --- 应用图像降采样 ---
            if downscale_factor > 1:
                new_width = size[0] // downscale_factor
                new_height = size[1] // downscale_factor
                imgL_rectified = cv2.resize(imgL_rectified_full, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                imgR_rectified = cv2.resize(imgR_rectified_full, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            else:
                imgL_rectified = imgL_rectified_full
                imgR_rectified = imgR_rectified_full

            # --- 计算视差图 (在降采样后的图像上) ---
            disparity = stereo.compute(imgL_rectified, imgR_rectified)
            if disparity is None:
                print("警告: SGBM 计算返回 None")
                # 使用上一帧有效的 threeD 或置空，避免后续错误
                # threeD = None # 或者保持上一帧的 threeD
                continue

            disparity_float = disparity.astype(np.float32) / 16.0
            # --- 使用调整后的 Q 矩阵进行三维重建 ---
            threeD = cv2.reprojectImageTo3D(disparity_float, Q_downscaled, handleMissingValues=True)
            # 注意：此时的 threeD 对应的是降采样后的图像尺寸

        except cv2.error as e:
            print(f"OpenCV 错误在处理帧时发生: {e}")
            threeD = None # 出错时重置 threeD
            continue
        except Exception as e:
            print(f"处理帧时发生未知错误: {e}")
            threeD = None # 出错时重置 threeD
            continue

        # --- 更新显示和鼠标回调 ---
        # 显示时可以使用降采样后的图像或全尺寸图像
        # 选项 A: 显示降采样后的图像 (更快)
        # imgL_display = cv2.cvtColor(imgL_rectified, cv2.COLOR_GRAY2BGR)
        # 选项 B: 显示全尺寸校正图像 (更清晰，但绘制可能稍慢)
        imgL_display = cv2.cvtColor(imgL_rectified_full, cv2.COLOR_GRAY2BGR)


        # --- 绘制辅助信息 (在全尺寸图像上绘制) ---
        # 注意：如果显示降采样图像，绘制坐标也需要相应缩放
        draw_coordinate_axes(imgL_display, transformer)
        visualize_ground_plane(imgL_display, transformer, left_camera_matrix, left_distortion)

        # 绘制采集的地面点 (投影到全尺寸图像)
        if len(ground_points) > 0:
            try:
                points_cam_vis = np.array(ground_points, dtype=np.float32).reshape(-1, 1, 3)
                points_img_vis, _ = cv2.projectPoints(points_cam_vis, np.zeros(3), np.zeros(3), left_camera_matrix, left_distortion)
                if points_img_vis is not None:
                    for pt in points_img_vis:
                        px, py = map(int, pt[0])
                        if 0 <= px < imgL_display.shape[1] and 0 <= py < imgL_display.shape[0]:
                            cv2.circle(imgL_display, (px, py), 3, (0, 255, 255), -1)
            except Exception as e:
                print(f"绘制地面点时出错: {e}")

        cv2.putText(imgL_display, f"FPS: {fps:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("left", imgL_display)

        # --- 传递 threeD 给回调函数 ---
        # 将降采样后的 threeD 和 downscale_factor 传递过去
        callback_param = {"threeD": threeD, "downscale_factor": downscale_factor}
        if threeD is not None:
             cv2.setMouseCallback("left", onmouse_pick_points, callback_param)
        else:
             cv2.setMouseCallback("left", onmouse_pick_points, {"threeD": None, "downscale_factor": downscale_factor})


        # --- 处理键盘输入 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("检测到 'q' 键，正在退出...")
            break

        # --- 其他按键处理 ('g', 'f', 'x', 'r') ---
        elif key == ord('g'):
             print("准备采集地面点：按住 Alt + 鼠标左键 在左视图中点击地面。")

        elif key == ord('f'): # 拟合地面
             if len(ground_points) >= 10:
                 try:
                     new_normal = fit_ground_plane(ground_points)
                     transformer.set_ground_normal(new_normal)
                     print(f"地面法向量已更新: {new_normal}")
                     print(f"新的旋转矩阵 R_cam_to_world:\n{transformer.R_cam_to_world}")
                 except ValueError as e:
                     print(f"拟合地面失败: {e}")
                 except Exception as e:
                     print(f"拟合地面时发生未知错误: {e}")
             else:
                 print(f"地面点不足 ({len(ground_points)}/10)，请继续采集 (Alt+左键)。")

        elif key == ord('x'): # 设置X轴
             # 使用全尺寸校正图像进行检测
             if imgL_rectified_full is None or imgR_rectified_full is None or threeD is None:
                 print("警告：无法设置X轴，因为当前帧处理不完整。")
                 continue

             print("尝试检测棋盘格以设置X轴...")
             # 在全尺寸图上检测，获取全尺寸像素坐标
             ret, corners_left_full, _ = detect_checkerboard(imgL_rectified_full, imgR_rectified_full)
             if not ret:
                 print("未检测到完整棋盘格！无法设置X轴。")
                 continue

             # 获取全尺寸像素坐标
             pt1_px_full = corners_left_full[0][0]
             pt2_px_full = corners_left_full[pattern_size[0] - 1][0]

             # --- 转换为降采样后的坐标以查询 threeD 图 ---
             pt1_coords_scaled = (int(pt1_px_full[0] / downscale_factor), int(pt1_px_full[1] / downscale_factor))
             pt2_coords_scaled = (int(pt2_px_full[0] / downscale_factor), int(pt2_px_full[1] / downscale_factor))

             height_scaled, width_scaled = threeD.shape[:2]
             valid_pt1 = 0 <= pt1_coords_scaled[1] < height_scaled and 0 <= pt1_coords_scaled[0] < width_scaled
             valid_pt2 = 0 <= pt2_coords_scaled[1] < height_scaled and 0 <= pt2_coords_scaled[0] < width_scaled

             if valid_pt1 and valid_pt2:
                 # 从降采样后的 threeD 图获取三维点
                 point1 = threeD[pt1_coords_scaled[1], pt1_coords_scaled[0]]
                 point2 = threeD[pt2_coords_scaled[1], pt2_coords_scaled[0]]

                 if (np.all(np.isfinite(point1)) and not np.any(np.abs(point1) > 1e6)) and \
                    (np.all(np.isfinite(point2)) and not np.any(np.abs(point2) > 1e6)):
                     ref_x_axis = point2 - point1
                     norm = np.linalg.norm(ref_x_axis)
                     if norm > 1e-6:
                         ref_x_axis /= norm
                         transformer.ref_x_axis = ref_x_axis
                         try:
                             transformer._compute_rotation()
                             print(f"标定板X轴方向已更新: {ref_x_axis}")
                             print(f"新的旋转矩阵 R_cam_to_world:\n{transformer.R_cam_to_world}")
                         except Exception as e:
                             print(f"更新旋转矩阵时出错: {e}")
                     else:
                         print("警告：计算标定板X轴失败 - 角点计算出的三维点重合或非常接近。")
                 else:
                     print("警告：标定板角点处的三维坐标无效 (深度值无效或过大)，无法设置X轴。")
             else:
                 print("警告：标定板角点像素坐标(缩放后)超出图像范围，无法获取三维坐标。")
             # --- 结束 'x' 键逻辑 ---

        elif key == ord('r'): # 重置地面点
             ground_points = []
             print("地面点列表已清空，可以重新采集 (按住Alt+左键)。")


    # --- 清理 ---
    print("正在停止...")
    frame_reader.stop()
    cv2.destroyAllWindows()
    print("已退出。")
