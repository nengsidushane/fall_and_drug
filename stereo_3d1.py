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


class CoordinateTransformer:
    def __init__(self):
        self.origin_cam = None  # 原点在相机坐标系中的坐标 (mm)
        self.R_cam_to_world = None  # 旋转矩阵
        self.ground_normal = None  # 地面法向量 (单位向量)

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
    global transform_origin, axis_points, transformer
    if event == cv2.EVENT_LBUTTONDOWN:
        imgL_rectified, imgR_rectified = param

        # 计算视差和三维坐标（原有代码）

        disparity = stereo.compute(imgL_rectified, imgR_rectified)
        disparity_float = disparity.astype(np.float32) / 16.0
        threeD = cv2.reprojectImageTo3D(disparity_float, Q, handleMissingValues=True)

        # 获取当前点的坐标（单位：毫米）
        point_cam = np.array([
            threeD[y][x][0],
            threeD[y][x][1],
            threeD[y][x][2]
        ])
        # 采集地面点（按住Alt键时触发）
        if flags & cv2.EVENT_FLAG_ALTKEY:
            # 深度过滤（仅采集1m-3m范围内的点）
            if 500 < np.linalg.norm(point_cam) < 3000:
                # 空间分布检查（与前5个点距离需大于50mm）
                if len(ground_points) < 5 or np.min(np.linalg.norm(ground_points - point_cam, axis=1)) > 50:
                    ground_points.append(point_cam)
                    print('添加地面点', point_cam[0] / 1000, point_cam[1] / 1000, point_cam[2] / 1000)

        # 按下'Shift'键时设置原点
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            if not np.all(np.isfinite(point_cam)) or np.any(np.abs(point_cam) > 1e6):
                print("警告：无效坐标，请重新选择原点")
                return
            transformer.set_origin(point_cam)
            print(f"原点已设置: {point_cam / 1000} m")

        # 按下'G'键时验证地面坐标系Z轴
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


# 全局初始化
ground_points = []
transformer = CoordinateTransformer()
# 地面法向量
ground_normal = np.array([0, 0, 1])
transformer.set_ground_normal(ground_normal)
#capture = cv2.VideoCapture(0)
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
set_ground = False


def calculate_point_operation(x, y, imgL_rectified, imgR_rectified):
    """
    三维坐标处理函数
    :param x: 图像x坐标
    :param y: 图像y坐标
    :param imgL_rectified: 左视图校正图像
    :param imgR_rectified: 右视图校正图像
    """
    global ground_points, transformer

    # 计算三维坐标
    disparity = stereo.compute(imgL_rectified, imgR_rectified)
    disparity_float = disparity.astype(np.float32) / 16.0
    threeD = cv2.reprojectImageTo3D(disparity_float, Q, handleMissingValues=True)

    # 获取当前点坐标（单位：毫米）
    point_cam = np.array([threeD[y][x][0],
                          threeD[y][x][1],
                          threeD[y][x][2]])
    # 坐标转换验证
    point_world = transformer.transform(point_cam)
    print(f"转换后坐标: {point_world / 1000} m")
    return point_world


def get_3d_coordinates(frame, ret, x, y):
    global set_ground
    if not set_ground:
        # 分割左右视图
        frame1 = frame[:, :1280]
        frame2 = frame[:, 1280:]

        # 校正图像
        imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        imgL_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
        imgR_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

        # 显示图像
        imgL_display = cv2.cvtColor(imgL_rectified, cv2.COLOR_GRAY2BGR)
        imgR_display = cv2.cvtColor(imgR_rectified, cv2.COLOR_GRAY2BGR)
        draw_coordinate_axes(imgL_display, transformer)
        return calculate_point_operation(x, y, imgL_rectified, imgR_rectified)
    else:
        while True:
            if not ret:
                continue

            # 分割左右视图
            frame1 = frame[:, :1280]
            frame2 = frame[:, 1280:]

            # 校正图像
            imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            imgL_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
            imgR_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

            # 显示图像
            imgL_display = cv2.cvtColor(imgL_rectified, cv2.COLOR_GRAY2BGR)
            imgR_display = cv2.cvtColor(imgR_rectified, cv2.COLOR_GRAY2BGR)
            draw_coordinate_axes(imgL_display, transformer)

            # 设置鼠标回调
            cv2.setMouseCallback("left", onmouse_pick_points, (imgL_rectified, imgR_rectified))

            # 显示图像
            cv2.imshow("left", imgL_display)

            # 处理键盘输入
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('g') and set_ground:  # 开始采集地面点
                ground_points = []
                print("开始采集地面点：按住Alt+左键点击地面区域")
            elif key == ord('f') and set_ground:  # 拟合平面
                try:
                    ground_normal = fit_ground_plane(ground_points)
                    print(f"地面法向量: [{ground_normal[0]:.6f}, {ground_normal[1]:.6f}, {ground_normal[2]:.6f}]")
                    print("法向量与垂直方向夹角：", np.degrees(np.arccos(np.dot(ground_normal, [0, -1, 0]))))
                    transformer.set_ground_normal(ground_normal)
                except Exception as e:
                    print(f"地面标定失败: {str(e)}")
            # 在鼠标回调中添加参考轴采集
            elif key == ord('x') and set_ground:  # 按X键采集标定板方向
                # 检测棋盘格
                ret, corners_left, corners_right = detect_checkerboard(imgL_rectified, imgR_rectified)
                if not ret:
                    print("未检测到完整棋盘格！")
                    continue

                # 计算三维坐标
                disparity = stereo.compute(imgL_rectified, imgR_rectified)
                disparity_float = disparity.astype(np.float32) / 16.0
                threeD = cv2.reprojectImageTo3D(disparity_float, Q)

                # 提取参考轴
                pt1 = (int(corners_left[0][0][0]), int(corners_left[0][0][1]))
                pt2 = (int(corners_left[pattern_size[0] - 1][0][0]), int(corners_left[pattern_size[0] - 1][0][1]))
                point1 = threeD[pt1[1], pt1[0]]
                point2 = threeD[pt2[1], pt2[0]]
                ref_x_axis = point2 - point1
                ref_x_axis /= np.linalg.norm(ref_x_axis)

                # 更新变换器
                transformer.ref_x_axis = ref_x_axis
                print(f"标定板X轴方向已设置: {ref_x_axis}")

            elif key == ord('r'):  # 按r键开启调整
                # 检测棋盘格
                set_ground = True

            elif key == ord('p'):  # 按p键关闭调整
                set_ground = False
