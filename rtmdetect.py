import numpy as np
import torch
from mmpose.apis import init_model
from mmpose.structures import PoseDataSample
from mmengine.dataset import Compose, pseudo_collate
import cv2
def rtmpose_inference_frame(
        frame: np.ndarray,
        bboxes_xyxy: list) -> list:
    """使用RTMPose模型在摄像头帧上进行人体关键点检测
    
    Args:
        frame (np.ndarray): 摄像头帧图像，形状为[H, W, C]
        bboxes_xyxy (list): 检测框列表，格式为[[x1,y1,x2,y2],...]
    
    Returns:
        list: 关键点检测结果，格式为[
            [[x1,y1,score1], [x2,y2,score2], ...],  # 第一个检测框的关键点
            [[x1,y1,score1], [x2,y2,score2], ...],  # 第二个检测框的关键点
            ...
        ]
    """
    # ================== 模型初始化 ==================
    # 硬编码模型配置文件和权重路径（根据实际路径修改）
    config_path = 'configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py'
    checkpoint_path = 'work_dirs/rtmpose-t_8xb256-420e_coco-256x192/best_published_coco_AP_epoch_420-2b6ae455_20250316.pth'
    
    # 使用缓存机制避免重复加载模型
    if not hasattr(rtmpose_inference_frame, 'model'):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        rtmpose_inference_frame.model = init_model(
            config_path, 
            checkpoint_path, 
            device=device
        )
    model = rtmpose_inference_frame.model
    
    # ================== 数据处理 ==================
    # 转换检测框格式为numpy数组
    bboxes = np.array(bboxes_xyxy, dtype=np.float32)
    
    # 获取预处理流水线
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    
    # 构建数据列表
    data_list = []
    for bbox in bboxes:
        data_info = {
            'img': frame,
            'bbox': bbox[None],  # 添加batch维度 -> shape (1,4)
            'bbox_score': np.ones(1, dtype=np.float32),
            'bbox_id': 0,  # 检测框ID
            'id': 0,       # 样本ID
            **model.dataset_meta
        }
        data_list.append(pipeline(data_info))
    
    # ================== 模型推理 ==================
    if data_list:
        batch = pseudo_collate(data_list)
        with torch.no_grad():
            results = model.test_step(batch)
    else:
        results = []
    
    # ================== 结果解析 ==================
    output = []
    for i, data_sample in enumerate(results):
        if isinstance(data_sample, PoseDataSample):
            # 获取原始检测框信息
            original_bbox = bboxes[i]
            
            # 获取预处理中的仿射变换矩阵
            warp_mat = data_sample.get('warp_mat', np.eye(3))
            
            # 反变换关键点坐标到原始图像空间
            keypoints = data_sample.pred_instances.keypoints[0]
            keypoints = cv2.perspectiveTransform(
                keypoints.reshape(-1, 1, 2).astype(np.float32), 
                np.linalg.inv(warp_mat)
            ).reshape(-1, 2)
            scores = data_sample.pred_instances.keypoint_scores[0]
        
            # 如果是PyTorch张量则转换
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
                keypoints = keypoints.cpu().numpy()
        
            # 组合成[x, y, score]格式
            kpts = np.concatenate([
                keypoints, 
                scores[:, None]
            ], axis=1).tolist()
            output.append(kpts)
    
    return output