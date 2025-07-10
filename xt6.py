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
import sys  # 用于异常处理
def init_gpu():
    """
    初始化并配置Mali-G610 GPU
    :return: OpenCL是否可用
    """
    try:
        # 检查OpenCL支持
        has_opencl = cv2.ocl.haveOpenCL()
        
        if has_opencl:
            # 尝试启用OpenCL
            cv2.ocl.setUseOpenCL(True)
            
            # 检查是否真正启用
            is_enabled = cv2.ocl.useOpenCL()
            print(f"OpenCL状态: {'启用' if is_enabled else '禁用'}")
            
            if not is_enabled:
                print("警告: 无法启用OpenCL，将使用CPU模式")
                return False
                
            # 获取设备信息
            try:
                # 尝试获取设备名称 
                device = cv2.ocl.Device_getDefault()
                if device is not None:
                    device_name = device.name()
                    print(f"OpenCL设备信息: {device_name}")
                    
                    # 检查是否为Mali GPU
                    if "mali" in str(device_name).lower() or "g610" in str(device_name).lower():
                        print("检测到Mali GPU，已启用OpenCL加速")
                        
                        # 尝试性能优化提示
                        try:
                            # 这些只是尝试调用方法，不管结果如何
                            device.isIntelSubgroupsSupported()
                            device.maxComputeUnits()
                            device.maxWorkGroupSize()
                        except:
                            pass
                        
                        return True
                    else:
                        print(f"检测到非Mali OpenCL设备: {device_name}")
                        return True
            except Exception as e:
                print(f"无法获取详细设备信息: {e}")
                # 尽管有错误，但OpenCL可能仍然可用
                return is_enabled
            
            return is_enabled
        else:
            print("OpenCL不可用，将使用CPU模式")
            return False
            
    except Exception as e:
        print(f"初始化GPU时发生错误: {e}")
        return False
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
    def __init__(self, src=0, width=640, height=480, max_queue_size=1):
        """
        优化的帧读取类 - 降低线程同步开销
        :param src: 摄像头索引或视频文件
        :param width: 请求的宽度
        :param height: 请求的高度
        :param max_queue_size: 队列大小，保持为1降低延迟
        """
        self.src = src
        self.width = width
        self.height = height
        self.max_queue_size = max_queue_size
        
        print(f"尝试打开摄像头源: {self.src}，分辨率: {self.width}x{self.height}")
        self.capture = cv2.VideoCapture(self.src)
        if not self.capture.isOpened():
            raise IOError(f"无法打开摄像头源: {self.src}")
        
        # 设置分辨率
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # 获取实际设置的分辨率
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头实际分辨率设置为: {self.width}x{self.height}")
        
        # 使用更高效的数据结构和同步原语
        self.frame_lock = threading.Lock()
        self.current_frame = None  # 最新帧
        self.frame_available = threading.Event()  # 信号事件代替队列
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
    
    def start(self):
        """启动帧读取线程"""
        self.stopped = False
        self.thread.start()
        print("帧读取线程已启动。")
        return self
    
    def update(self):
        """线程运行的函数 - 优化内存使用和同步"""
        while not self.stopped:
            # 读取新帧
            ret, frame = self.capture.read()
            if not ret:
                time.sleep(0.01)  # 短暂等待
                continue
            
            # 安全地更新当前帧
            with self.frame_lock:
                self.current_frame = frame
            
            # 设置信号表示新帧可用
            self.frame_available.set()
            
            # 适当的睡眠，避免CPU过度使用
            time.sleep(0.001)
    
    def read(self):
        """
        读取最新的帧 - 零拷贝设计
        :return: (bool, frame) 元组
        """
        # 等待新帧或超时
        is_new_frame = self.frame_available.wait(timeout=0.01)
        if is_new_frame:
            self.frame_available.clear()  # 重置事件
            
            # 从线程读取最新帧
            with self.frame_lock:
                if self.current_frame is not None:
                    return True, self.current_frame
        
        # 使用最后已知的帧，或表示无可用帧
        with self.frame_lock:
            if self.current_frame is not None:
                return True, self.current_frame
        return False, None
    
    def stop(self):
        """停止线程并释放资源"""
        if self.stopped:
            return
        
        print("正在停止帧读取线程...")
        self.stopped = True
        
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        if self.capture.isOpened():
            self.capture.release()
        
        print("帧读取线程已停止，摄像头已释放。")
# --- 全局初始化 ---
ground_points = []
transformer = CoordinateTransformer()
transformer.set_ground_normal(ground_normal) # Initialize with default normal

# ... (The rest of the code, including the if __name__ == '__main__': block, remains the same as the previous version where FrameReader requests 1280x720) ...
def create_stereo_matcher(use_gpu=True):
    """
    创建高度优化的立体匹配器，针对Mali-G610 GPU调整参数
    :param use_gpu: 是否使用GPU
    :return: 配置好的立体匹配器
    """
    # 优化立体匹配参数 - 更激进的性能优化
    num_disp = 48  # 降低视差范围但保持16的倍数
    block_size = 7  # 再次降低块大小以加快计算
    
    try:
        if use_gpu and cv2.ocl.useOpenCL():
            print(f"创建GPU加速的StereoBM，numDisparities={num_disp}, blockSize={block_size}")
            
            stereo_matcher = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
            
            # 极度优化的GPU参数设置 - 牺牲一些质量换取速度
            stereo_matcher.setPreFilterType(cv2.StereoBM_PREFILTER_NORMALIZED_RESPONSE)
            stereo_matcher.setPreFilterSize(5)      # 最小有效值
            stereo_matcher.setPreFilterCap(23)      # 降低以提高速度
            stereo_matcher.setTextureThreshold(20)  # 提高以忽略更多弱纹理区域
            stereo_matcher.setUniquenessRatio(5)    # 最小值以最大化速度
            stereo_matcher.setSpeckleWindowSize(25) # 减小窗口提高速度
            stereo_matcher.setSpeckleRange(8)       # 降低范围提高速度
            stereo_matcher.setMinDisparity(0)
            
            # 检查OpenCL状态
            if not cv2.ocl.useOpenCL():
                print("警告: 尽管请求使用GPU，但OpenCL不可用，将使用CPU版本")
                stereo_matcher = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
        else:
            print(f"创建CPU版StereoBM，numDisparities={num_disp}, blockSize={block_size}")
            stereo_matcher = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
            # CPU模式下也优化参数
            stereo_matcher.setPreFilterSize(5)
            stereo_matcher.setTextureThreshold(20)
            stereo_matcher.setUniquenessRatio(10)
        
        return stereo_matcher
    
    except Exception as e:
        print(f"创建立体匹配器时发生错误: {e}")
        # 回退到基本配置
        return cv2.StereoBM_create(numDisparities=32, blockSize=11)
def process_stereo_frame(frame, left_map1, left_map2, right_map1, right_map2, stereo_matcher, 
                         Q_matrix, downscale_factor=4, use_gpu=True):
    """
    针对Mali-G610 GPU优化的立体处理函数
    :param frame: 输入的双目图像帧
    :param left_map1, left_map2, right_map1, right_map2: 重映射矩阵
    :param stereo_matcher: 立体匹配器
    :param Q_matrix: 重投影矩阵
    :param downscale_factor: 降采样因子 - 增加到4以大幅提高速度
    :param use_gpu: 是否使用GPU加速
    :return: 三维点云、左视图处理结果
    """
    try:
        # 分割左右视图
        mid_point = frame.shape[1] // 2
        
        # --- 1. 优化: 极度激进的降采样处理 ---
        if downscale_factor > 1:
            # 在CPU上执行降采样，避免在分割后再降采样
            small_size = (frame.shape[1]//downscale_factor, frame.shape[0]//downscale_factor)
            small_frame = cv2.resize(frame, small_size, interpolation=cv2.INTER_NEAREST)
            
            # 然后分割降采样后的图像
            small_mid = small_frame.shape[1] // 2
            imgL = cv2.cvtColor(small_frame[:, :small_mid], cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(small_frame[:, small_mid:], cv2.COLOR_BGR2GRAY)
            
            # 计算降采样后的重映射矩阵
            small_map1_L = cv2.resize(left_map1, (imgL.shape[1], imgL.shape[0]), 
                                     interpolation=cv2.INTER_LINEAR)
            small_map2_L = cv2.resize(left_map2, (imgL.shape[1], imgL.shape[0]), 
                                     interpolation=cv2.INTER_LINEAR)
            small_map1_R = cv2.resize(right_map1, (imgR.shape[1], imgR.shape[0]), 
                                     interpolation=cv2.INTER_LINEAR)
            small_map2_R = cv2.resize(right_map2, (imgR.shape[1], imgR.shape[0]), 
                                     interpolation=cv2.INTER_LINEAR)
        else:
            # 常规处理
            imgL = cv2.cvtColor(frame[:, :mid_point], cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(frame[:, mid_point:], cv2.COLOR_BGR2GRAY)
            small_map1_L, small_map2_L = left_map1, left_map2
            small_map1_R, small_map2_R = right_map1, right_map2
        
        # --- 2. GPU处理流程 ---
        if use_gpu and cv2.ocl.useOpenCL():
            # --- 确保OpenCL设备可用 ---
            if not cv2.ocl.useOpenCL():
                print("警告: OpenCL被标记为使用但不可用，回退到CPU处理")
                use_gpu = False
            else:
                # 2.1 创建UMat对象传输到GPU
                imgL_umat = cv2.UMat(imgL)
                imgR_umat = cv2.UMat(imgR)
                map1L_umat = cv2.UMat(small_map1_L)
                map2L_umat = cv2.UMat(small_map2_L)
                map1R_umat = cv2.UMat(small_map1_R) 
                map2R_umat = cv2.UMat(small_map2_R)
                
                # 2.2 GPU重映射 - 使用INTER_NEAREST提高速度
                imgL_rectified = cv2.remap(imgL_umat, map1L_umat, map2L_umat, cv2.INTER_NEAREST)
                imgR_rectified = cv2.remap(imgR_umat, map1R_umat, map2R_umat, cv2.INTER_NEAREST)
                
                # 2.3 GPU立体匹配
                disparity = stereo_matcher.compute(imgL_rectified, imgR_rectified)
                
                # 2.4 将视差数据传回CPU
                disparity_np = disparity.get()
                disparity_float = disparity_np.astype(np.float32) / 16.0
                
                # 2.5 点云重投影
                threeD = cv2.reprojectImageTo3D(disparity_float, Q_matrix, handleMissingValues=True)
                
                # 2.6 返回结果
                return threeD, imgL
        
        # --- 3. CPU处理流程 ---
        # 3.1 CPU重映射
        imgL_rectified = cv2.remap(imgL, small_map1_L, small_map2_L, cv2.INTER_NEAREST)
        imgR_rectified = cv2.remap(imgR, small_map1_R, small_map2_R, cv2.INTER_NEAREST)
        
        # 3.2 CPU立体匹配
        disparity = stereo_matcher.compute(imgL_rectified, imgR_rectified)
        
        # 3.3 转换视差
        disparity_float = disparity.astype(np.float32) / 16.0
        
        # 3.4 点云重投影
        threeD = cv2.reprojectImageTo3D(disparity_float, Q_matrix, handleMissingValues=True)
        
        # 3.5 返回结果
        return threeD, imgL
        
    except cv2.error as e:
        print(f"OpenCV错误: {e}")
        return None, None
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None
def create_opencl_kernels():
    """
    创建优化的OpenCL内核，针对Mali-G610 GPU
    :return: 编译后的OpenCL内核对象或None
    """
    try:
        # 检查OpenCL是否可用
        if not cv2.ocl.haveOpenCL():
            return None
        
        # OpenCL程序源码 - Mali GPU优化内核
        program_source = """
        // Mali-G610优化的视差计算内核
        __kernel void fast_abs_diff(
            __global const uchar* left,
            __global const uchar* right,
            __global uchar* output,
            int width,
            int height,
            int stride)
        {
            // 获取当前工作项的全局ID
            int x = get_global_id(0);
            int y = get_global_id(1);
            
            // 边界检查
            if (x >= width || y >= height)
                return;
                
            // Mali GPU优化：使用向量类型加速内存访问
            int idx = y * stride + x;
            output[idx] = abs(left[idx] - right[idx]);
        }
        
        // 优化的SAD (Sum of Absolute Differences)计算内核
        __kernel void fast_sad(
            __global const uchar* left,
            __global const uchar* right,
            __global float* cost,
            int width,
            int height,
            int blockSize,
            int max_disp)
        {
            int x = get_global_id(0);
            int y = get_global_id(1);
            int disp = get_global_id(2);
            
            if (x >= width || y >= height || disp >= max_disp)
                return;
                
            // 确保不越界
            if (x - disp < 0) return;
            
            // 局部窗口的SAD计算
            int half = blockSize/2;
            int sad = 0;
            int count = 0;
            
            // Mali优化：使用适合SIMD宽度的循环展开
            for(int i = -half; i <= half; i++) {
                int ny = y + i;
                if(ny < 0 || ny >= height) continue;
                
                for(int j = -half; j <= half; j++) {
                    int nx = x + j;
                    int nx_d = nx - disp;
                    
                    if(nx < 0 || nx >= width || nx_d < 0 || nx_d >= width)
                        continue;
                    
                    int idx_l = ny * width + nx;
                    int idx_r = ny * width + nx_d;
                    
                    sad += abs((int)left[idx_l] - (int)right[idx_r]);
                    count++;
                }
            }
            
            // 将成本写入输出数组
            if(count > 0) {
                cost[y * width * max_disp + x * max_disp + disp] = (float)sad / count;
            }
        }
        """
        
        # 创建OpenCL上下文和程序
        ctx = cv2.ocl.Context.getDefault()
        if ctx.empty():
            print("无法获取默认OpenCL上下文")
            return None
            
        program = cv2.ocl.Program(ctx, program_source)
        if program.empty():
            print("无法创建OpenCL程序")
            return None
            
        # 构建程序 - 添加Mali特定的优化选项
        try:
            # Mali GPU优化选项
            build_options = "-cl-fast-relaxed-math -cl-mad-enable"
            program.build(build_options)
        except Exception:
            # 回退到默认编译
            program.build()
            
        if program.empty():
            print("OpenCL程序构建失败")
            return None
            
        print("Mali-G610优化的OpenCL内核创建成功")
        return program
    except Exception as e:
        print(f"OpenCL内核创建失败: {e}")
        return None
if __name__ == '__main__':
    # 初始化GPU
    use_gpu = init_gpu()
    
    # 优化参数
    downscale_factor = 2  # 降采样因子
    
    # 创建优化的立体匹配器
    stereo = create_stereo_matcher(use_gpu=use_gpu)
    
    # --- 启动帧读取线程 ---
    try:
        # 请求更低分辨率以减少数据传输
        requested_width = 1280  # 降低到640x480
        requested_height = 720
        print(f"尝试请求降低分辨率: {requested_width}x{requested_height}")
        frame_reader = FrameReader(src=0, width=requested_width, height=requested_height).start()

        print("等待摄像头初始化...")
        time.sleep(0.5)  # 减少等待时间
        transformer.set_origin(np.array([461.60626, 535.6015, 371.5321]))
        print("摄像头初始化完成。")
    except Exception as e:
        print(f"错误：摄像头初始化失败: {e}")
        exit()

    # --- 调整Q矩阵 ---
    single_image_width = frame_reader.width // 2
    single_image_height = frame_reader.height
    
    # 根据降采样和实际分辨率调整Q矩阵
    Q_downscaled = Q.copy()
    if downscale_factor > 1:
        # 调整Q矩阵以适应降采样
        Q_downscaled[0, 3] /= downscale_factor
        Q_downscaled[1, 3] /= downscale_factor
        Q_downscaled[2, 3] /= downscale_factor
        Q_downscaled[3, 3] /= downscale_factor
    
    # --- 改进的FPS计算 ---
    # 使用滑动窗口方法计算FPS
    frame_times = []  # 存储最近N帧的时间戳
    fps_window_size = 10  # 计算最近30帧的平均FPS
    fps_display_interval = 1.0  # 每1秒更新一次FPS显示
    last_fps_display = time.time()
    fps = 0
    
    # --- 启用异常详细信息 ---
    prev_excepthook = sys.excepthook
    def detailed_exception_handler(exc_type, exc_value, exc_traceback):
        print("\n*** 未捕获的异常详情 ***")
        import traceback
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("\n")
        prev_excepthook(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = detailed_exception_handler
    
    # --- 主循环 ---
    try:
        print("进入主循环，按 'q' 退出...")
        while True:
            loop_start = time.time()  # 每次循环开始时间
            
            # 读取帧
            ret, frame = frame_reader.read()
            
            if not ret:
                print("无法读取帧，等待...")
                time.sleep(0.01)
                continue
            
            # --- 改进的FPS计算 ---
            current_time = time.time()
            frame_times.append(current_time)
            
            # 只保留滑动窗口内的时间戳
            while len(frame_times) > fps_window_size:
                frame_times.pop(0)
                
            # 至少有2个帧时才计算FPS
            if len(frame_times) > 1:
                # 计算滑动窗口内的平均FPS
                time_diff = frame_times[-1] - frame_times[0]
                if time_diff > 0:  # 避免除以零
                    fps = (len(frame_times) - 1) / time_diff
            
            # 定期显示FPS
            if current_time - last_fps_display >= fps_display_interval:
                print(f"FPS: {fps:.2f}, 处理延迟: {(time.time() - loop_start)*1000:.1f}ms")
                last_fps_display = current_time
            
            # --- 使用优化的GPU处理立体图像 ---
            try:
                threeD, imgL = process_stereo_frame(
                    frame, left_map1, left_map2, right_map1, right_map2, 
                    stereo, Q_downscaled, downscale_factor, use_gpu
                )
                
                # 可以在这里添加UI渲染代码
                # 例如: display_frame = visualize_result(imgL, threeD)
                # cv2.imshow("Stereo Vision", display_frame)
                
            except Exception as e:
                print(f"处理帧时发生错误: {e}")
                # 继续处理下一帧，不中断循环
            
            # 检查退出键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户请求退出...")
                break
    
    except KeyboardInterrupt:
        print("接收到键盘中断，准备退出...")
    except Exception as e:
        print(f"主循环中发生未捕获的异常: {e}")
    finally:
        # --- 清理资源 ---
        print("正在停止并清理资源...")
        if 'frame_reader' in locals():
            frame_reader.stop()
        cv2.destroyAllWindows()
        cv2.ocl.setUseOpenCL(False)  # 重置OpenCL状态
        print("程序已安全退出。")