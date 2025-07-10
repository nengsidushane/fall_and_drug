import numpy as np
from enum import Enum
from typing import List, Tuple, Dict, Optional, Union

class MedicationPosture(Enum):
    """服药姿态枚举类"""
    NO_ACTION = 0           # 无动作
    APPROACHING_MOUTH = 1   # 手接近嘴部
    TAKING_MEDICINE = 2     # 正在服药
    AFTER_TAKING = 3        # 服药后

def recognize_medication_posture(
    hand_keypoints: np.ndarray,  # 形状为 [3]，手部的三维坐标 (x, y, z)
    mouth_position: np.ndarray,  # 形状为 [3]，嘴部的三维坐标 (x, y, z)
    medicine_positions: List[np.ndarray]  # 长度为 3 的列表，每个元素是一种药品的三维坐标 (x, y, z)
) -> Dict:

    # 处理手部关键点，确定掌心位置
    hand_keypoints = np.array(hand_keypoints)
    palm_center = hand_keypoints

    
    # 计算手掌中心到嘴部的距离
    hand_to_mouth_distance = np.linalg.norm(palm_center - mouth_position)
    
    # 计算手掌中心到各个药品的距离
    medicine_distances = [np.linalg.norm(palm_center - med_pos) for med_pos in medicine_positions]
    
    # 找到最近的药品及其索引
    closest_medicine_idx = np.argmin(medicine_distances)
    closest_medicine_distance = medicine_distances[closest_medicine_idx]
    
    HAND_MEDICINE_THRESHOLD = 0.1  # 手与药品接触的距离阈值
    HAND_MOUTH_CLOSE_THRESHOLD = 0.15  # 手靠近嘴部的距离阈值
    HAND_MOUTH_TOUCH_THRESHOLD = 0.05  # 手与嘴接触的距离阈值
    #姿态判断：
    #首先判断手是否拿着药品（通过 HAND_MEDICINE_THRESHOLD 判断）
    #如果拿着药品：
    #当手与嘴距离小于 HAND_MOUTH_TOUCH_THRESHOLD 时，判定为"正在服药"
    #当手与嘴距离小于 HAND_MOUTH_CLOSE_THRESHOLD 时，判定为"准备服药"
    #如果没拿药品但手接触嘴部，判定为"服药后"
    # 识别姿态
    result = {
        "posture": MedicationPosture.NO_ACTION,
        "medicine_index": None,
        "confidence": 0.0
    }
    
    # 1. 检查手是否拿着药品
    holding_medicine = closest_medicine_distance < HAND_MEDICINE_THRESHOLD
    
    # 2. 判断姿态
    if holding_medicine:
        if hand_to_mouth_distance < HAND_MOUTH_TOUCH_THRESHOLD:
            # 手拿着药物并且接触到嘴部 - 正在服药
            result["posture"] = MedicationPosture.TAKING_MEDICINE
            result["medicine_index"] = closest_medicine_idx
            # 距离越小，置信度越高
            result["confidence"] = 1.0 - hand_to_mouth_distance / HAND_MOUTH_TOUCH_THRESHOLD
        elif hand_to_mouth_distance < HAND_MOUTH_CLOSE_THRESHOLD:
            # 手拿着药物并且靠近嘴部 - 准备服药
            result["posture"] = MedicationPosture.APPROACHING_MOUTH
            result["medicine_index"] = closest_medicine_idx
            # 根据距离计算置信度
            confidence_range = HAND_MOUTH_CLOSE_THRESHOLD - HAND_MOUTH_TOUCH_THRESHOLD
            distance_in_range = hand_to_mouth_distance - HAND_MOUTH_TOUCH_THRESHOLD
            result["confidence"] = 1.0 - (distance_in_range / confidence_range)
    elif hand_to_mouth_distance < HAND_MOUTH_TOUCH_THRESHOLD:
        # 手接触嘴部但没有药品 - 可能是服药后
        result["posture"] = MedicationPosture.AFTER_TAKING
        result["confidence"] = 1.0 - hand_to_mouth_distance / HAND_MOUTH_TOUCH_THRESHOLD
    
    # 确保置信度在有效范围内
    result["confidence"] = max(0.0, min(1.0, result["confidence"]))
    
    return result

# 示例使用
if __name__ == "__main__":
    single_keypoint = np.array([0.45, 0.45, 0.45])  # 单个掌心三维坐标
    mouth_position = np.array([0.5, 0.5, 0.5])#转成array格式数组
    medicine_positions = [
        np.array([0.2, 0.2, 0.2]),  # 药品1
        np.array([0.8, 0.8, 0.8]),  # 药品2
        np.array([0.4, 0.4, 0.4])   # 药品3
    ]
    
    # 使用单个掌心点识别姿态
    result = recognize_medication_posture(single_keypoint, mouth_position, medicine_positions)
    print(result)
    print(result["posture"])

'''

输入：
single_keypoint: 手部的三维坐标 (x, y, z)
mouth_position: 嘴部的三维坐标 (x, y, z)
medicine_positions: 三种药品的三维坐标 (x, y, z)
函数：
    recognize_medication_posture(
        hand_keypoints: np.ndarray,
        mouth_position: np.ndarray,
        medicine_positions: List[np.ndarray]
    )
输出 -> dict
"posture": 服药姿态
"medicine_index": 药品索引
"confidence": 置信度

'''