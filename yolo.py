import os

import cv2
import numpy as np
from base import RK3588


class RK3588_v2(RK3588):
    NMS_THRESH = 0.75
    OBJ_THRESH = 0.05
    # 在 RK3588_v2 类中添加新方法
    def process_with_model(self, rknn_model, frame, accelerate):
        """使用提供的RKNN模型进行推理处理
        
        Args:
            rknn_model: 来自rknnpool的RKNN模型实例
            frame: 需要处理的帧
            accelerate: 加速度参数
            
        Returns:
            处理后的帧和检测框
        """
        # 保存原始尺寸
        orig_img_h, orig_img_w = frame.shape[:2]
        
        # 预处理图像
        img = self.preprocess(frame)
        
        # 使用传入的模型进行推理，而不是自带的rknn实例
        outputs = rknn_model.inference(inputs=[img])
        
        # 后处理结果
        result_frame, boxes = self.postprocess(outputs, frame, accelerate)
        
        return result_frame, boxes
    def pre_process(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        img, ratio, dwdh = self.letterbox(img, auto=False)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0).astype(np.float32)

        return img, ratio, dwdh

    def inference(self, img: np.ndarray) -> list[np.ndarray] | None:
        #print(self.rknn.inference(inputs=[img]))
        return self.rknn.inference(inputs=[img])

    def filter_boxes(
        self,
        boxes: np.ndarray,
        box_confidences: np.ndarray,
        box_class_probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter boxes with object threshold."""
        box_confidences = box_confidences.flatten()
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        scores = class_max_score * box_confidences
        mask = scores >= self.OBJ_THRESH

        return boxes[mask], classes[mask], scores[mask]

    def dfl(self, position: np.ndarray) -> np.ndarray:
        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num
        y = position.reshape(n, p_num, mc, h, w)

        exp_y = np.exp(y)
        y = exp_y / np.sum(exp_y, axis=2, keepdims=True)

        acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(float)
        return np.sum(y * acc_metrix, axis=2)

    def box_process(self, position: np.ndarray) -> np.ndarray:
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
        grid = np.stack((col, row), axis=0).reshape(1, 2, grid_h, grid_w)
        stride = np.array([self.net_size // grid_h, self.net_size // grid_w]).reshape(
            1, 2, 1, 1
        )

        position = self.dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

        return xyxy

    def post_process(
        self, outputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        def sp_flatten(_in):
            ch = _in.shape[1]
            return _in.transpose(0, 2, 3, 1).reshape(-1, ch)

        defualt_branch = 3
        pair_per_branch = len(outputs) // defualt_branch

        boxes, classes_conf, scores = [], [], []
        for i in range(defualt_branch):
            boxes.append(self.box_process(outputs[pair_per_branch * i]))
            classes_conf.append(sp_flatten(outputs[pair_per_branch * i + 1]))
            scores.append(np.ones_like(classes_conf[-1][:, :1], dtype=np.float32))

        boxes = np.concatenate([sp_flatten(b) for b in boxes])
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores).flatten()

        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self.OBJ_THRESH, self.NMS_THRESH
        )
        if isinstance(indices, tuple):
            return None, None, None

        indices = [i for i in indices if scores[i] > self.OBJ_THRESH]

        boxes = boxes[indices]
        classes = classes[indices]
        scores = scores[indices]

        #print(boxes)

        return boxes, classes, scores


