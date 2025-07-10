# rtmpose_rknn.py
import numpy as np
import cv2
from rknn.api import RKNN
import logging
from typing import Tuple
import argparse
import time
from typing import List, Tuple

import cv2
import loguru
import numpy as np
import onnxruntime as ort
from rknn.api import RKNN

from stereo_3d import calculate_point_operation


def init_rknn_model(model_path: str) -> RKNN:
    """Initialize RKNN model from file."""
    rknn = RKNN()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise ValueError(f"Load RKNN model failed, error code: {ret}")
    ret = rknn.init_runtime(target='rk3588',core_mask=RKNN.NPU_CORE_ALL)
    if ret != 0:
        raise ValueError(f"Init runtime failed, error code: {ret}")
    return rknn

def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """Extend the scale to match the given aspect ratio."""
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(w > h * aspect_ratio,
                          np.hstack([w, w / aspect_ratio]),
                          np.hstack([h * aspect_ratio, h]))
    return bbox_scale

def rtmpose_inference_frame(
    frame: np.ndarray,
    bboxes_xyxy: list
) -> list:
    """使用RKNN模型进行人体关键点检测
    
    Args:
        frame (np.ndarray): 摄像头帧图像，形状为[H, W, C]
        bboxes_xyxy (list): 检测框列表，格式为[[x1,y1,x2,y2],...]
    
    Returns:
        list: 关键点检测结果，格式为[
            [[x1,y1,score1], [x2,y2,score2], ...],  # 第一个检测框的关键点 17
            [[x1,y1,score1], [x2,y2,score2], ...],  # 第二个检测框的关键点 17
            ...
        ]
    """
    # ================== 模型初始化 ==================
    if not hasattr(rtmpose_inference_frame, 'rknn'):
        model_path = 'models/rtmposes_int8.rknn'  # 硬编码模型路径
        rtmpose_inference_frame.rknn = init_rknn_model(model_path)
    rknn = rtmpose_inference_frame.rknn

    # 强制将None转换为空列表
    bboxes_xyxy = bboxes_xyxy if bboxes_xyxy is not None else []

    # ================== 预处理和推理 ==================
    output_list = []
    model_input_size = (192, 256)  # 模型输入尺寸（w, h）
    
    for bbox in bboxes_xyxy:
        x1, y1, x2, y2 = map(int, bbox[:4])

        # 检查坐标有效性
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)  # 原图宽度
        y2 = min(frame.shape[0], y2)  # 原图高度

        if x2 <= x1 or y2 <= y1:
            print(f"无效检测框: {bbox}, 跳过处理")
            continue

        crop_img = frame[y1:y2, x1:x2]
        
        # 仿射变换预处理
        resized_img, center, scale = preprocess(crop_img, model_input_size)
        
        # 推理
        outputs = rknn.inference(inputs=[resized_img.astype(np.float16)])
        simcc_x, simcc_y = outputs[0], outputs[1]
        
        # 后处理
        keypoints, scores = postprocess([simcc_x, simcc_y], model_input_size, center, scale)
        
        # 转换到原图坐标系
        keypoints[:, 0] += x1
        keypoints[:, 1] += y1
        output_list.append(np.hstack([keypoints, scores[:, None]]).tolist())

        if output_list is []:
            return [np.empty((0, 17, 3))]  # 标准三维空数组
    return output_list


import numpy as np


def extract_specific_keypoints(
        keypoints: np.ndarray,
        scores: np.ndarray,
        thr: float = 0.3
) -> dict:
    """
    提取鼻子和脚部关键点（基于COCO关键点定义）

    参数:
        keypoints (np.ndarray): 形状为(N, K, 2)的关键点坐标数组
        scores (np.ndarray): 形状为(N, K)的置信度分数数组
        thr (float): 置信度阈值，默认0.3

    返回:
        dict: 包含过滤后关键点的字典，结构示例：
            {
                "nose": [x, y],
                "left_foot": [x, y],
                "right_foot": [x, y]
            }

    参考索引定义[1](@ref):
        - 鼻子: 0
        - 左脚踝: 15
        - 右脚踝: 16
    """
    # COCO关键点索引定义
    KEYPOINT_INDEX = {
        "nose": 0,
        "left_foot": 15,
        "right_foot": 16
    }

    selected = {}
    for name, idx in KEYPOINT_INDEX.items():
        instance_idx = np.argmax(scores[:, idx])

        if scores[instance_idx][idx] > thr:
            coord = keypoints[instance_idx][idx].tolist()
            rounded_coord = [round(c, 2) for c in coord]
            selected[name] = [rounded_coord]  # 包装成二维数组
        else:
            selected[name] = []  # 无效时返回空二维数组

    return selected

def get_pose_coordinates(body_keypoints, left_frame, right_frame):
    """返回格式：[[nose_x, nose_y, nose_z], [feet_x, feet_y, feet_z]]"""
    default_coord = [0.0, 0.0, 0.0]  # 无效坐标默认值

    # 获取鼻子坐标
    nose_coord = body_keypoints.get('nose', [])
    nose_3d = default_coord
    if len(nose_coord) > 0 and len(nose_coord[0]) == 2:
        nose_result = calculate_point_operation(nose_coord, left_frame, right_frame)
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
        feet_result = calculate_point_operation(feet_coord, left_frame, right_frame)
        if feet_result and feet_result[0][0] != -1:
            feet_3d = feet_result[0]

    return [nose_3d, feet_3d]

# 以下为原RKNN代码中的依赖函数
def preprocess(img: np.ndarray, input_size: tuple) -> tuple:
    """预处理函数（与原代码相同）"""
    img_shape = img.shape[:2]
    bbox = np.array([0, 0, img_shape[1], img_shape[0]])
    center, scale = bbox_xyxy2cs(bbox, padding=1.25)
    resized_img, _ = top_down_affine(input_size, scale, center, img)
    return resized_img, center, scale

def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.) -> tuple:
    """坐标转换（与原代码相同）"""
    dim = bbox.ndim
    if dim == 1: bbox = bbox[None, :]
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding
    return (center[0], scale[0]) if dim == 1 else (center, scale)

def top_down_affine(input_size: tuple, scale: np.ndarray, center: np.ndarray, img: np.ndarray) -> tuple:
    """仿射变换（与原代码相同）"""
    w, h = input_size
    bbox_scale = _fix_aspect_ratio(scale, aspect_ratio=w/h)
    warp_mat = get_warp_matrix(center, bbox_scale, 0, (w, h))
    resized_img = cv2.warpAffine(img, warp_mat, (w, h), flags=cv2.INTER_LINEAR)
    return resized_img, bbox_scale

def postprocess(outputs: list, model_input_size: tuple, center: tuple, scale: tuple) -> tuple:
    """后处理（与原代码相同）"""
    simcc_x, simcc_y = outputs
    keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio=2.0)
    keypoints = keypoints / model_input_size * scale + center - scale / 2
    return keypoints[0], scores[0]


def get_warp_matrix(center: np.ndarray,
                    scale: np.ndarray,
                    rot: float,
                    output_size: Tuple[int, int],
                    shift: Tuple[float, float] = (0., 0.),
                    inv: bool = False) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # compute transformation matrix
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    # get four corners of the src rectangle in the original image
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    # get four corners of the dst rectangle in the input image
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat

def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c

def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt

def decode(simcc_x: np.ndarray, simcc_y: np.ndarray,
           simcc_split_ratio) -> Tuple[np.ndarray, np.ndarray]:
    """Modulate simcc distribution with Gaussian.

    Args:
        simcc_x (np.ndarray[K, Wx]): model predicted simcc in x.
        simcc_y (np.ndarray[K, Wy]): model predicted simcc in y.
        simcc_split_ratio (int): The split ratio of simcc.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: keypoints in shape (K, 2) or (n, K, 2)
        - np.ndarray[float32]: scores in shape (K,) or (n, K)
    """
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio

    return keypoints, scores

def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # get maximum value locations
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # get maximum value across x and y axis
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    # reshape
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals