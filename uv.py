import cv2
import numpy as np
import time
import math
from sklearn.linear_model import RANSACRegressor

# 初始化相机参数和校正参数
left_camera_matrix = np.array([[650.8727267, 0, 632.0660297],
                               [0, 651.5738202, 358.6525714],
                               [0, 0, 1]])
right_camera_matrix = np.array([[653.5158803, 0, 642.6118299],
                                [0, 653.7522913, 353.6209308],
                                [0, 0, 1]])
left_distortion = np.array([[0, 0, 0, 0, 0]])
right_distortion = np.array([[0, 0, 0, 0, 0]])
R = np.array([[0.999921585, 0.001239856, -0.012461435],
              [-0.001141365, 0.999968082, 0.007907702],
              [0.012470841, -0.007892859, 0.999891085]])
T = np.array([[-61.59406398], [0.140562835], [-0.721770621]])
size = (1280, 720)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)


def onmouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        print('\n像素坐标 x = %d, y = %d' % (x, y))
        print("世界坐标xyz 是：", threeD[y][x][0] / 1000.0, threeD[y][x][1] / 1000.0, threeD[y][x][2] / 1000.0, "m")
        distance = math.sqrt(threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2) / 1000.0
        print("距离是：", distance, "m")


capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280 * 2)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = capture.read()
    if not ret: break

    frame1 = frame[0:720, 0:1280]
    frame2 = frame[0:720, 1280:2560]

    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

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

    disparity = stereo.compute(img1_rectified, img2_rectified)
    disp_actual = disparity.astype(np.float32) / 16.0

    # 生成V视差图
    valid_mask = (disp_actual >= 0) & (disp_actual <= 128)
    valid_v, valid_u = np.where(valid_mask)
    valid_d = np.round(disp_actual[valid_v, valid_u]).astype(int)

    v_disparity = np.zeros((disp_actual.shape[0], 129), dtype=np.float32)  # 改为float32
    np.add.at(v_disparity, (valid_v, valid_d), 1)

    # 添加高斯平滑
    v_disparity = cv2.GaussianBlur(v_disparity, (5, 5), 0)

    # 归一化并阈值处理
    v_disp_norm = cv2.normalize(v_disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, thresh = cv2.threshold(v_disp_norm, 50, 255, cv2.THRESH_BINARY)

    # 生成U视差图（横向统计）
    u_disparity = np.zeros((disp_actual.shape[1], 129), dtype=np.float32)
    valid_u_indices = np.where(valid_mask)
    np.add.at(u_disparity, (valid_u_indices[1], valid_d), 1)
    u_disparity = cv2.GaussianBlur(u_disparity, (5, 5), 0)

    # 检测U视差图中的垂直线（障碍物）
    u_disp_norm = cv2.normalize(u_disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, u_thresh = cv2.threshold(u_disp_norm, 30, 255, cv2.THRESH_BINARY)
    u_lines = cv2.HoughLinesP(u_thresh, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    # 提取候选点并拟合RANSAC模型
    points = np.argwhere(thresh > 0)
    if len(points) > 1:
        X = points[:, 0].reshape(-1, 1)  # v坐标作为特征
        y = points[:, 1]  # d值作为目标
        try:
            ransac = RANSACRegressor(random_state=42)
            ransac.fit(X, y)
            a = ransac.estimator_.coef_[0]
            b = ransac.estimator_.intercept_

            # 生成地面掩膜
            ground_mask = np.zeros_like(disp_actual, dtype=np.uint8)
            for v in range(disp_actual.shape[0]):
                d_pred = a * v + b
                if 0 <= d_pred <= 128:
                    lower = max(0, int(d_pred - 2))
                    upper = min(128, int(d_pred + 2))
                    mask_row = (disp_actual[v, :] >= lower) & (disp_actual[v, :] <= upper)
                    ground_mask[v, mask_row] = 1

            # 在原图上绘制地面区域
            overlay = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
            overlay[ground_mask == 1] = [0, 255, 0]
            # 根据U视差图过滤垂直障碍物
            if u_lines is not None:
                for line in u_lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x1 - x2) < 5:  # 垂直线特征
                        d = x1
                        ground_mask[:, int(d - 2):int(d + 2)] = 0  # 剔除垂直障碍物区域

            # 深度约束：限制最大检测距离（例如5米）
            max_depth = 10  # 5米对应的视差值需根据Q矩阵计算
            ground_mask[disp_actual > max_depth] = 0
        except:
            overlay = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    else:
        overlay = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)

    # 形态学闭操作填充空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, kernel)

    # 连通域分析，剔除小区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ground_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 500:
            ground_mask[labels == i] = 0

    # 显示结果
    cv2.imshow('Ground Detection', overlay)
    cv2.imshow('V Disparity', cv2.resize(v_disp_norm, (400, 720)))
    cv2.imshow('U Disparity', cv2.resize(u_disp_norm, (400, 720)))

    dis_color = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True) * 16
    # cv2.imshow("depth", dis_color)
    cv2.setMouseCallback("depth", onmouse_pick_points, threeD)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
