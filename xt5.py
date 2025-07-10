import cv2
import numpy as np
from matplotlib.pyplot import gray
from sklearn.linear_model import LinearRegression, RANSACRegressor
import threading
import queue
import time # For sleep
import threading
import os # To get CPU count
import cProfile
import pstats
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
# We'll request 1280x720 from FrameReader later, so single image size is 640x720
size = (640, 720) # Single image size for rectification

# --- Calculate rectification maps based on the single image size ---
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                            right_camera_matrix, right_distortion, size, R, T)
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
pattern_size = (11, 8)
img_channels = 3

# --- Switch to StereoBM ---
# Note: StereoBM parameters often require different tuning than SGBM
# numDisparities must be divisible by 16
num_disp_bm = 64   # Try 64, adjust based on results (must be > 0 and divisible by 16)
block_size_bm = 15 # Must be odd, adjust based on results (e.g., 5, 7, ..., 21)

print(f"使用 StereoBM，numDisparities={num_disp_bm}, blockSize={block_size_bm}")
stereo = cv2.StereoBM_create(numDisparities=num_disp_bm, blockSize=block_size_bm)
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
    鼠标回调函数 - Optimized for downscaling
    :param param: 字典，应包含 'threeD' 点云图和 'downscale_factor',
                  e.g., {"threeD": threeD_map, "downscale_factor": factor}
    """
    global ground_points, transformer
    if event == cv2.EVENT_LBUTTONDOWN:
        if "threeD" not in param or param["threeD"] is None:
            # print("警告：回调函数缺少有效的'threeD'点云图") # 减少打印
            return
        if "downscale_factor" not in param:
            print("错误：回调函数缺少 'downscale_factor'")
            return

        threeD = param["threeD"]
        downscale_factor = param["downscale_factor"]

        # --- 将鼠标点击的全尺寸坐标 (x, y) 转换为降采样后的坐标 ---
        x_scaled = x // downscale_factor
        y_scaled = y // downscale_factor

        # 检查缩放后的坐标是否在降采样后的 threeD 图范围内
        if 0 <= y_scaled < threeD.shape[0] and 0 <= x_scaled < threeD.shape[1]:
            # 从降采样后的三维点云图中获取坐标 (单位：毫米)
            point_cam = threeD[y_scaled, x_scaled]

            # 检查获取的坐标是否有效
            if not np.all(np.isfinite(point_cam)) or np.any(np.abs(point_cam) > 1e6):
                # print("警告：选定点的深度值无效或超出范围，请重新选择。") # 减少打印
                return

            # --- 后续逻辑与之前类似，使用有效的 point_cam ---
            if flags & cv2.EVENT_FLAG_ALTKEY:
                depth = np.linalg.norm(point_cam)
                if 500 < depth < 3000:
                    is_far_enough = True
                    if len(ground_points) >= 5:
                         min_dist = np.min(np.linalg.norm(np.array(ground_points[-5:]) - point_cam, axis=1))
                         if min_dist <= 50:
                             is_far_enough = False
                    if is_far_enough:
                        ground_points.append(point_cam)
                        print(f'添加地面点 {len(ground_points)}: {point_cam / 1000} m')
                    # else: # 减少打印
                    #     print("警告：新点离最近的5个点太近 (<50mm)")
                # else: # 减少打印
                #     print(f"警告：点深度 {depth:.1f} mm 不在有效范围 (500-3000mm)")

            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                transformer.set_origin(point_cam)
                print(f"原点已设置: {point_cam / 1000} m")

            elif flags & cv2.EVENT_FLAG_CTRLKEY:
                try:
                    point_world = transformer.transform(point_cam)
                    print(f"相机坐标: {point_cam / 1000} m -> 世界坐标: {point_world / 1000} m")
                except ValueError as e:
                    print(f"坐标转换错误: {e}")
                except Exception as e:
                    print(f"处理点击时发生未知错误: {e}")
        # else: # 减少打印
        #     print(f"警告：点击坐标(缩放后) ({x_scaled}, {y_scaled}) 超出图像范围 ({threeD.shape[1]}x{threeD.shape[0]})")



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

import cv2
import numpy as np
# ... other imports ...
import threading
import os # To get CPU count

# ... (previous code, including CoordinateTransformer, FrameReader, etc.) ...

def fit_ground_plane_worker(points, num_iterations, inlier_threshold, result_list, thread_id):
    """RANSAC worker function for a single thread."""
    local_best_normal = None
    local_max_inliers = -1 # Use -1 to ensure any valid result is better
    n_points = len(points)

    if n_points < 3:
        result_list[thread_id] = (None, -1)
        return

    # Use a thread-safe random number generator if necessary,
    # but np.random.choice is generally okay for read-only data access.
    # rng = np.random.default_rng() # Alternative if needed

    for _ in range(num_iterations):
        try:
            # sample_indices = rng.choice(n_points, 3, replace=False)
            sample_indices = np.random.choice(n_points, 3, replace=False)
            sample = points[sample_indices]

            # --- Plane fitting ---
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)

            # Avoid degenerate cases (collinear points)
            if norm_n < 1e-6:
                continue
            n /= norm_n
            # --- End Plane fitting ---

            # --- Inlier counting ---
            # Vectorized distance calculation
            dist = np.abs(np.dot(points - sample[0], n))
            inliers = np.sum(dist < inlier_threshold)
            # --- End Inlier counting ---

            # --- Update best model (local to thread) ---
            if inliers > local_max_inliers:
                local_max_inliers = inliers
                local_best_normal = n
            # --- End Update ---

        except Exception as e:
            # Catch potential errors during sampling or calculation within a thread
            print(f"[Thread {thread_id}] Error during RANSAC iteration: {e}")
            continue # Skip this iteration

    # Store the best result found by this thread
    result_list[thread_id] = (local_best_normal, local_max_inliers)


def fit_ground_plane(points, num_total_iterations=100, inlier_threshold=5.0, num_threads=None):
    """
    PCA初筛 + 多线程RANSAC精修
    :param points: Nx3 numpy array of points.
    :param num_total_iterations: Total number of RANSAC iterations across all threads.
    :param inlier_threshold: RANSAC inlier distance threshold (mm).
    :param num_threads: Number of threads to use. Defaults to CPU count.
    :return: Best fit ground normal vector or None if fitting fails.
    """
    points = np.array(points, dtype=np.float32)
    n_points = len(points)

    if n_points < 3:
        print("警告: fit_ground_plane 需要至少 3 个点。")
        return None # Return None instead of raising error to avoid crashing main loop

    # --- 1. 初步PCA拟合 (remains the same) ---
    try:
        mean, eigvec = cv2.PCACompute(points, mean=None)
        # Check if PCACompute returned valid eigenvectors
        if eigvec is None or len(eigvec) < 3:
             print("警告: PCACompute未能计算有效的特征向量。")
             return None
        normal_pca = eigvec[2]
        # Normalize PCA normal just in case
        norm_pca = np.linalg.norm(normal_pca)
        if norm_pca < 1e-6:
            print("警告: PCA计算出的法向量接近零向量。")
            return None
        normal_pca /= norm_pca

    except cv2.error as e:
        print(f"错误: OpenCV PCACompute 失败: {e}")
        return None
    except Exception as e:
        print(f"错误: PCA 初筛时发生未知错误: {e}")
        return None


    # --- 2. 多线程 RANSAC ---
    if num_threads is None:
        num_threads = os.cpu_count()
        if num_threads is None: # Fallback if cpu_count fails
            num_threads = 4
        num_threads = max(1, num_threads) # Ensure at least one thread

    # Distribute iterations among threads
    iterations_per_thread = num_total_iterations // num_threads
    extra_iterations = num_total_iterations % num_threads

    threads = []
    results = [None] * num_threads # Pre-allocate list to store results

    print(f"使用 {num_threads} 个线程进行 RANSAC ({num_total_iterations} 总迭代)...")
    for i in range(num_threads):
        iters = iterations_per_thread + (1 if i < extra_iterations else 0)
        if iters == 0: continue # Skip if no iterations assigned

        thread = threading.Thread(target=fit_ground_plane_worker,
                                  args=(points, iters, inlier_threshold, results, i))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # --- 3. 收集结果并找到最佳模型 ---
    best_normal_overall = normal_pca # Initialize with PCA result
    max_inliers_overall = -1

    for result in results:
        if result is not None:
            normal, inliers = result
            if normal is not None and inliers > max_inliers_overall:
                max_inliers_overall = inliers
                best_normal_overall = normal

    if max_inliers_overall == -1:
        print("警告: RANSAC 未能找到任何有效的平面模型，将使用PCA结果。")
        # best_normal_overall remains normal_pca

    # --- 4. 方向校正 (假设相机Y轴向下或根据场景调整) ---
    # Ensure the final normal points generally upwards relative to the camera's typical orientation
    # If the camera's Y-axis points downwards in the world, a positive Y component
    # in the normal means it's pointing downwards.
    if best_normal_overall[1] > 0:
        best_normal_overall *= -1

    # Final normalization just to be safe
    final_norm = np.linalg.norm(best_normal_overall)
    if final_norm < 1e-6:
        print("警告: 最终计算出的法向量接近零向量，返回PCA结果。")
        # Recalculate PCA normal direction if needed
        if normal_pca[1] > 0:
             normal_pca *= -1
        return normal_pca
    else:
        best_normal_overall /= final_norm


    print(f"RANSAC 完成。最佳模型内点数: {max_inliers_overall}/{n_points}")
    return best_normal_overall


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


def visualize_ground_plane(img, transformer, camera_matrix, dist_coeffs, plane_size=2000, grid_step=1000): # Increased grid_step (was 500)
    """
    在图像上绘制地面网格 - Optimized Density
    :param img: 原始图像（BGR格式）
    :param transformer: CoordinateTransformer实例
    :param camera_matrix: 相机内参矩阵
    :param dist_coeffs: 畸变系数
    :param plane_size: 地面网格尺寸（毫米），默认2m x 2m
    :param grid_step: 网格线间隔（毫米），默认1000mm (1 meter) - Reduced density
    """
    if transformer.origin_cam is None or transformer.R_cam_to_world is None:
        return img

    # 生成网格点（世界坐标系，Z=0） - Fewer points due to larger grid_step
    x_range = np.arange(-plane_size // 2, plane_size // 2 + 1, grid_step)
    y_range = np.arange(-plane_size // 2, plane_size // 2 + 1, grid_step)
    # Ensure at least 2 points per dimension for drawing lines
    if len(x_range) < 2: x_range = np.linspace(-plane_size // 2, plane_size // 2, 2)
    if len(y_range) < 2: y_range = np.linspace(-plane_size // 2, plane_size // 2, 2)

    xx, yy = np.meshgrid(x_range, y_range)
    zz = np.zeros_like(xx)
    points_world = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # Check if points_world is empty (should not happen with the check above)
    if points_world.shape[0] == 0:
        return img

    # 转换到相机坐标系
    try:
        R_world_to_cam = transformer.R_cam_to_world.T
        # Ensure origin_cam is compatible for broadcasting/addition
        origin_cam_reshaped = transformer.origin_cam.reshape(1, 3)
        points_cam = (R_world_to_cam @ points_world.T).T + origin_cam_reshaped
    except Exception as e:
        print(f"Error during coordinate transformation in visualize_ground_plane: {e}")
        return img


    # 投影到图像平面
    try:
        points_img, _ = cv2.projectPoints(
            points_cam.astype(np.float32),
            np.zeros(3), np.zeros(3), # Use zeros for rvec, tvec as points_cam is already in camera coords
            camera_matrix, dist_coeffs
        )
        if points_img is None: # Check if projection failed
             print("Warning: cv2.projectPoints returned None in visualize_ground_plane.")
             return img
        points_img = points_img.reshape(-1, 2).astype(int)
    except cv2.error as e:
        print(f"OpenCV error during projection in visualize_ground_plane: {e}")
        return img
    except Exception as e:
        print(f"Error during projection in visualize_ground_plane: {e}")
        return img


    # 绘制网格线
    color = (0, 255, 0)  # 绿色
    n_rows, n_cols = xx.shape # Use shape of the meshgrid

    img_height, img_width = img.shape[:2]

    # Draw horizontal lines
    for i in range(n_rows):
        for j in range(n_cols - 1):
            idx1 = i * n_cols + j
            idx2 = i * n_cols + j + 1
            pt1 = tuple(points_img[idx1])
            pt2 = tuple(points_img[idx2])
            # Basic clipping check (can be improved with cv2.clipLine)
            if 0 <= pt1[0] < img_width and 0 <= pt1[1] < img_height and \
               0 <= pt2[0] < img_width and 0 <= pt2[1] < img_height:
                cv2.line(img, pt1, pt2, color, 1)

    # Draw vertical lines
    for j in range(n_cols):
        for i in range(n_rows - 1):
            idx1 = i * n_cols + j
            idx2 = (i + 1) * n_cols + j
            # Ensure idx2 is within bounds (already implicitly handled by loop range)
            # if idx2 < len(points_img):
            pt1 = tuple(points_img[idx1])
            pt2 = tuple(points_img[idx2])
            # Basic clipping check
            if 0 <= pt1[0] < img_width and 0 <= pt1[1] < img_height and \
               0 <= pt2[0] < img_width and 0 <= pt2[1] < img_height:
                cv2.line(img, pt1, pt2, color, 1)

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
    def __init__(self, src=0, width=1280, height=720, max_queue_size=1):
        """
        初始化 FrameReader。
        :param src: 摄像头源索引或视频文件路径。
        :param width: 期望的摄像头宽度。
        :param height: 期望的摄像头高度。
        :param max_queue_size: 帧队列的最大尺寸。
        """
        self.src = src
        self.width = width
        self.height = height
        self.max_queue_size = max_queue_size

        print(f"尝试打开摄像头源: {self.src}，分辨率: {self.width}x{self.height}")
        self.capture = cv2.VideoCapture(self.src)
        if not self.capture.isOpened():
            raise IOError(f"无法打开摄像头源: {self.src}")

        # 尝试设置分辨率
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # 读取实际设置的分辨率
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头实际分辨率设置为: {actual_width}x{actual_height}")
        if actual_width != self.width or actual_height != self.height:
             print(f"警告: 未能将摄像头分辨率精确设置为 {self.width}x{self.height}。")
             # 更新 width 和 height 以匹配实际值，避免后续问题
             self.width = actual_width
             self.height = actual_height


        # 使用 LIFO 队列，当队列满时自动丢弃旧帧
        self.frame_queue = queue.LifoQueue(maxsize=self.max_queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # 设置为守护线程，主程序退出时自动结束

    def start(self):
        """启动帧读取线程。"""
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

            # LIFO Queue logic: Try to remove oldest if full, then put newest
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait() # Discard the oldest frame
                except queue.Empty:
                    pass # Should not happen if full() is true, but safety first
            try:
                # Use non-blocking put to avoid waiting here
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                # If it became full again between check and put (unlikely), drop frame
                pass

            # Avoid sleeping here unless absolutely necessary for CPU usage
            # time.sleep(0.001) # Keep commented out unless reader thread hogs CPU

    def read(self):
        """
        从队列中获取最新的帧。
        :return: (bool, frame) 元组，表示是否成功获取帧以及帧本身。
                 如果队列为空或超时，返回 (False, None)。
        """
        try:
            # --- Significantly reduce timeout ---
            # Wait only a very short time (e.g., 10ms)
            # If no frame is ready, main loop can continue quickly
            return True, self.frame_queue.get(timeout=0.01) # Reduced timeout
        except queue.Empty:
            # This is now expected frequently if processing is faster than reading
            return False, None # Indicate failure or timeout

    def stop(self):
        """停止读取线程并释放摄像头资源。"""
        if self.stopped: # Avoid stopping multiple times
            return
        print("正在停止帧读取线程...")
        self.stopped = True
        # 等待线程结束（设置短暂超时）
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.capture.isOpened():
            self.capture.release()
        print("帧读取线程已停止，摄像头已释放。")
# --- 全局初始化 ---
ground_points = []
transformer = CoordinateTransformer()
transformer.set_ground_normal(ground_normal) # Initialize with default normal

# ... (The rest of the code, including the if __name__ == '__main__': block, remains the same as the previous version where FrameReader requests 1280x720) ...

if __name__ == '__main__':
    # --- 启动帧读取线程 ---
    try:
        # --- Keep requesting the lower resolution ---
        requested_width = 1280 # Total width for side-by-side
        requested_height = 720
        print(f"尝试请求较低分辨率: {requested_width}x{requested_height}")
        frame_reader = FrameReader(src=0, width=requested_width, height=requested_height, max_queue_size=1).start()
        # --- End resolution change ---

        print("等待摄像头初始化...")
        time.sleep(1.5)
        transformer.set_origin(np.array([461.60626, 535.6015, 371.5321])) # Example origin
        print("摄像头初始化完成。")
    except IOError as e:
        print(f"错误：无法初始化摄像头: {e}")
        exit()
    except Exception as e:
        print(f"错误：启动 FrameReader 时发生未知错误: {e}")
        exit()

    # --- Adjust size variable for rectification based on ACTUAL single image size ---
    single_image_width = frame_reader.width // 2
    single_image_height = frame_reader.height
    # IMPORTANT: Check if the size used for map calculation matches the actual single image size
    if size != (single_image_width, single_image_height):
        print(f"警告: 初始尺寸 {size} 与 FrameReader 报告的单眼尺寸 {(single_image_width, single_image_height)} 不匹配!")
        print("需要确保校正映射与实际处理的图像尺寸一致。")
        # Ideally, recalculate maps here if they differ, but we did it above based on expectation.
        size = (single_image_width, single_image_height) # Update size variable just in case

    print(f"校正/处理将基于单眼图像尺寸: {size}")
    # Maps were already calculated above using the expected size (640, 720)

    # FPS 计算变量
    processed_frame_count = 0
    fps = 0
    start_time = cv2.getTickCount()
    last_fps_print_time = start_time
    fps_print_interval = 3.0

    # --- 优化: 帧降采样 (跳帧处理) ---
    frame_skip = 1
    fetch_count = 0

    # --- 优化: 图像降采样因子 ---
    downscale_factor = 2 # Keep downscaling factor
    if downscale_factor > 1:
        Q_downscaled = Q.copy()
        Q_downscaled[0, 3] /= downscale_factor
        Q_downscaled[1, 3] /= downscale_factor
        Q_downscaled[2, 3] /= downscale_factor
        Q_downscaled[3, 3] /= downscale_factor
        print(f"使用降采样因子 {downscale_factor}，已调整Q矩阵。处理尺寸: {single_image_width//downscale_factor}x{single_image_height//downscale_factor}")
    else:
        Q_downscaled = Q
        print(f"未使用降采样。处理尺寸: {single_image_width}x{single_image_height}")

    # --- 可视化控制标志 ---
    show_axes = True
    show_grid = True
    show_ground_points = True
    ground_points_update_counter = 0
    ground_points_update_interval = 5

    # ... (rest of the main loop: variable pre-allocation, loop start, profiler, frame reading, processing, visualization, key handling, cleanup) ...
    # The core processing logic inside the loop remains the same, but will now use the StereoBM object.

    print("开始主循环...")
    print("按 '1' 切换坐标轴显示")
    print("按 '2' 切换地面网格显示")
    print("按 '3' 切换采集点显示")
    print("按 'q' 退出")

    profiler = cProfile.Profile()
    profiler.enable()

    # while True:
    for i in range(1000): # Limit iterations for profiling run
        ret, frame = frame_reader.read()

        if not ret:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("检测到 'q' 键 (在等待帧时)，正在退出...")
                break
            continue

        fetch_count += 1
        if fetch_count % frame_skip != 0:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        processed_frame_count += 1

        # --- Calculate and Display FPS ---
        current_tick = cv2.getTickCount()
        elapsed_seconds_total = (current_tick - start_time) / cv2.getTickFrequency()
        if elapsed_seconds_total >= 1.0:
            fps = processed_frame_count / elapsed_seconds_total
            elapsed_since_last_print = (current_tick - last_fps_print_time) / cv2.getTickFrequency()
            if elapsed_since_last_print >= fps_print_interval:
                print(f"当前帧率: {fps:.2f} FPS")
                last_fps_print_time = current_tick
            start_time = current_tick
            processed_frame_count = 0

        # --- Computation Steps ---
        try:
            mid_point = frame_reader.width // 2
            frame1 = frame[:, :mid_point]
            frame2 = frame[:, mid_point:]

            imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            imgL_rectified_full = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_NEAREST)
            imgR_rectified_full = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_NEAREST)

            if downscale_factor > 1:
                new_width = size[0] // downscale_factor
                new_height = size[1] // downscale_factor
                imgL_rectified = cv2.resize(imgL_rectified_full, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                imgR_rectified = cv2.resize(imgR_rectified_full, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            else:
                imgL_rectified = imgL_rectified_full
                imgR_rectified = imgR_rectified_full

            # --- Compute disparity using StereoBM ---
            disparity = stereo.compute(imgL_rectified, imgR_rectified)
            if disparity is None:
                print("警告: stereo.compute 计算返回 None")
                continue

            # BM output needs normalization/scaling differently than SGBM for reprojectImageTo3D
            # It's often int16, scale by 1/16.0 like SGBM
            disparity_float = disparity.astype(np.float32) / 16.0

            threeD = cv2.reprojectImageTo3D(disparity_float, Q_downscaled, handleMissingValues=True)

        except cv2.error as e:
            print(f"OpenCV 错误在处理帧时发生: {e}")
            threeD = None
            continue
        except Exception as e:
            print(f"处理帧时发生未知错误: {e}")
            threeD = None
            continue

        # --- Visualization and UI ---
        imgL_display = cv2.cvtColor(imgL_rectified_full, cv2.COLOR_GRAY2BGR)

        if show_axes: draw_coordinate_axes(imgL_display, transformer)
        if show_grid: visualize_ground_plane(imgL_display, transformer, left_camera_matrix, left_distortion)

        ground_points_update_counter += 1
        if show_ground_points and ground_points_update_counter >= ground_points_update_interval:
            ground_points_update_counter = 0
            if len(ground_points) > 0:
                try:
                    points_cam_vis = np.array(ground_points, dtype=np.float32).reshape(-1, 1, 3)
                    points_img_vis, _ = cv2.projectPoints(points_cam_vis, np.zeros(3), np.zeros(3), left_camera_matrix, left_distortion)
                    if points_img_vis is not None:
                        for pt in points_img_vis:
                            px, py = map(int, pt[0])
                            if 0 <= px < imgL_display.shape[1] and 0 <= py < imgL_display.shape[0]:
                                cv2.circle(imgL_display, (px, py), 3, (0, 255, 255), -1)
                except Exception as e: print(f"绘制地面点时出错: {e}")
        elif show_ground_points: pass

        cv2.putText(imgL_display, f"FPS: {fps:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        status_text = f"Axes:{show_axes} Grid:{show_grid} Pts:{show_ground_points}"
        cv2.putText(imgL_display, status_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.imshow("left", imgL_display)

        # --- Mouse Callback ---
        callback_param = {"threeD": threeD, "downscale_factor": downscale_factor}
        cv2.setMouseCallback("left", onmouse_pick_points, callback_param)

        # --- Process Keyboard Input ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('1'): show_axes = not show_axes; print(f"坐标轴显示: {show_axes}")
        elif key == ord('2'): show_grid = not show_grid; print(f"地面网格显示: {show_grid}")
        elif key == ord('3'): show_ground_points = not show_ground_points; ground_points_update_counter = 0; print(f"采集点显示: {show_ground_points}")
        elif key == ord('g'): print("准备采集地面点：按住 Alt + 鼠标左键 在左视图中点击地面。")
        elif key == ord('f'):
             if len(ground_points) >= 10:
                 try:
                     new_normal = fit_ground_plane(ground_points, num_total_iterations=100)
                     if new_normal is not None: transformer.set_ground_normal(new_normal); print(f"地面法向量已更新: {new_normal}")
                     else: print("拟合地面失败，未能计算出有效的法向量。")
                 except Exception as e: print(f"拟合地面时发生未知错误: {e}")
             else: print(f"地面点不足 ({len(ground_points)}/10)，请继续采集 (Alt+左键)。")
        elif key == ord('x'):
             if imgL_rectified_full is None or imgR_rectified_full is None or threeD is None: print("警告：无法设置X轴，因为当前帧处理不完整。"); continue
             print("尝试检测棋盘格以设置X轴...")
             ret, corners_left_full, _ = detect_checkerboard(imgL_rectified_full, imgR_rectified_full)
             if not ret: print("未检测到完整棋盘格！无法设置X轴。"); continue
             pt1_px_full = corners_left_full[0][0]; pt2_px_full = corners_left_full[pattern_size[0] - 1][0]
             pt1_coords_scaled = (int(pt1_px_full[0] / downscale_factor), int(pt1_px_full[1] / downscale_factor))
             pt2_coords_scaled = (int(pt2_px_full[0] / downscale_factor), int(pt2_px_full[1] / downscale_factor))
             if threeD is not None:
                 height_scaled, width_scaled = threeD.shape[:2]
                 valid_pt1 = 0 <= pt1_coords_scaled[1] < height_scaled and 0 <= pt1_coords_scaled[0] < width_scaled
                 valid_pt2 = 0 <= pt2_coords_scaled[1] < height_scaled and 0 <= pt2_coords_scaled[0] < width_scaled
                 if valid_pt1 and valid_pt2:
                     point1 = threeD[pt1_coords_scaled[1], pt1_coords_scaled[0]]; point2 = threeD[pt2_coords_scaled[1], pt2_coords_scaled[0]]
                     if (np.all(np.isfinite(point1)) and not np.any(np.abs(point1) > 1e6)) and (np.all(np.isfinite(point2)) and not np.any(np.abs(point2) > 1e6)):
                         ref_x_axis = point2 - point1; norm = np.linalg.norm(ref_x_axis)
                         if norm > 1e-6:
                             ref_x_axis /= norm; transformer.ref_x_axis = ref_x_axis
                             try: transformer._compute_rotation(); print(f"标定板X轴方向已更新: {ref_x_axis}")
                             except Exception as e: print(f"更新旋转矩阵时出错: {e}")
                         else: print("警告：计算标定板X轴失败 - 角点计算出的三维点重合或非常接近。")
                     else: print("警告：标定板角点处的三维坐标无效 (深度值无效或过大)，无法设置X轴。")
                 else: print("警告：标定板角点像素坐标(缩放后)超出图像范围，无法获取三维坐标。")
             else: print("警告: threeD 图无效，无法设置X轴。")
        elif key == ord('r'): ground_points = []; print("地面点列表已清空，可以重新采集 (按住Alt+左键)。")

    # --- Profiler End & Report ---
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    print("\n--- Profiler Results (Top 20 Cumulative Time) ---")
    stats.print_stats(20)
    print("-------------------------------------------------")

    # --- Cleanup ---
    print("正在停止...")
    frame_reader.stop()
    cv2.destroyAllWindows()
    print("已退出。")
