# test_rtmpose.py
import cv2
import numpy as np

from accelerate import xyz_accelerate
from stereo_3d import calculate_point_operation
from simple_3d import compute_point_3d
from yolortmpose import rtmpose_inference_frame, extract_specific_keypoints


def visualize(img: np.ndarray,
              keypoints: list,
              filename: str = 'output_pose.jpg',
              thr: float = 0.3) -> np.ndarray:
    """可视化关键点和骨架
    
    Args:
        img: 原始BGR图像(H, W, 3)
        keypoints: 关键点列表[[[x1,y1,s1],...],...]
        filename: 保存路径
        thr: 可视化置信度阈值
    """
    # 定义绘制参数
    skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                (0, 1), (0, 2), (1, 3), (2, 4)]
    palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]]
    link_color = [1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0]
    point_color = [0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

    # 遍历每个检测框的关键点
    for kpts in keypoints:
        # 转换为numpy数组
        kpts_arr = np.array(kpts)  # shape (17,3)
        points = kpts_arr[:, :2]    # 坐标 (17,2)
        scores = kpts_arr[:, 2]    # 置信度 (17,)
        
        # 绘制关键点
        for j, (x, y) in enumerate(points):
            color_idx = point_color[j] if j < len(point_color) else 0
            cv2.circle(img, (int(x), int(y)), 4, palette[color_idx], -1)
        
        # 绘制骨架连线
        for link_idx, (u, v) in enumerate(skeleton):
            if u < len(scores) and v < len(scores):
                color_idx = link_color[link_idx] if link_idx < len(link_color) else 0
                if scores[u] > thr and scores[v] > thr:
                    cv2.line(img, 
                            (int(points[u][0]), int(points[u][1])),
                            (int(points[v][0]), int(points[v][1])),
                            palette[color_idx], 2)
    
    # 保存可视化结果
    cv2.imwrite(filename, img)
    return img

def get_pose_coordinates(body_keypoints, left_frame, right_frame):
    """返回格式：[[nose_x, nose_y, nose_z], [feet_x, feet_y, feet_z]]"""
    default_coord = [0.0, 0.0, 0.0]  # 无效坐标默认值

    # 获取鼻子坐标
    nose_coord = body_keypoints.get('nose', [])
    nose_3d = default_coord
    if len(nose_coord) > 0 and len(nose_coord[0]) == 2:
        # nose_result = calculate_point_operation(nose_coord, left_frame, right_frame)
        nose_result = compute_point_3d(nose_coord, left_frame, right_frame)
        #print("nose_result", nose_result)
        if nose_result and nose_result[0][0] != -1:
            nose_3d = nose_result[0]

    # 获取脚部坐标（优先左，其次右）
    feet_coord = []
    if len(body_keypoints.get('left_foot', [])) > 0:
        feet_coord = body_keypoints['left_foot']
    elif len(body_keypoints.get('right_foot', [])) > 0:
        feet_coord = body_keypoints['right_foot']

    feet_3d = default_coord
    if len(feet_coord) > 0 and len(feet_coord[0]) == 2:
        # feet_result = calculate_point_operation(feet_coord, left_frame, right_frame)
        feet_result = compute_point_3d(feet_coord, left_frame, right_frame)
        if feet_result and feet_result[0][0] != -1:
            feet_3d = feet_result[0]
    #print("nose_3d, feet_3d",[nose_3d, feet_3d])
    return [nose_3d, feet_3d]

def rtm(left_frame, right_frame, BBOXES):
    
    # 调用推理函数
    keypoints = rtmpose_inference_frame(left_frame, BBOXES)
    #print(keypoints)
    # 打印结果
    #print("关键点检测结果：")
    #for i, kpts in enumerate(keypoints):
        #print(f"检测框 {i+1}:")
        #for j, (x, y, score) in enumerate(kpts):
            #print(f"  关键点 {j+1}: x={x:.2f}, y={y:.2f}, score={score:.4f}")
    
    # 可视化并保存结果
    left_frame = visualize(left_frame, keypoints, 'visualized_output.jpg')
    # 提取关键点（可自定义阈值）
    #keypoints = rtmpose_inference_frame(left_frame, BBOXES)
    # 转换为NumPy三维数组（N人物×K关键点×3坐标）
    keypoints = np.asarray(keypoints)

    pos = [[],[]]

    if keypoints.ndim == 3:
        scores = keypoints[:, :, 2]

        body_keypoints = extract_specific_keypoints(
            keypoints=keypoints[:, :, :2],  # 仅坐标部分
            scores=scores,  # 正确的二维置信度数组
            thr=0.35
        )

        # 获取目标格式数据
        #print("body_keypoints",body_keypoints)
        pos = [0,0,0]
        pos = get_pose_coordinates(body_keypoints, left_frame, right_frame)#筛选出鼻子和腿
        
        # 示例输出：[[1.2, 2.3, 1.5], [0.8, 0.0, 1.1]]
        # print("POS:", pos)
        #print("accelerate:", xyz_accelerate(pos))

    #print("可视化结果已保存至 visualized_output.jpg")
    #return left_frame, pos, xyz_accelerate(pos)
    return left_frame

if __name__ == "__main__":
    main()