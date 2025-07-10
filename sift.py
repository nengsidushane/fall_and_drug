import cv2
import numpy as np
from align_transform import Align


def computer(img1, img2):
    """直接处理内存中的图像数组"""
    align = Align(img1, img2, threshold=1)
    similarity, _ = align.compute_similarity(use_ssim=False)
    print("similarity:")
    print(similarity)
    return similarity

if __name__ == "__main__":
    source = 'Images/img5.jpg'
    target = 'Images/img6.jpg'
    computer(source, target)
