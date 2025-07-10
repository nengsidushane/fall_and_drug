import cv2
import numpy as np


def orb_feature_matching(image1, image2):
    # 创建ORB检测器
    orb = cv2.ORB_create()

    # 在两张图像中检测ORB特征和计算描述符
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # 使用暴力匹配器（Brute-Force Matcher）进行匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # 进行特征点匹配
    matches = bf.match(des1, des2)

    # 根据匹配的距离排序
    matches = sorted(matches, key=lambda x: x.distance)
    # 可视化前N个匹配
    #img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 计算相似度作为返回值
    # 计算前N个匹配点的平均距离
    good_match_count = len(matches)

    avg_distance = np.mean([match.distance for match in matches[:good_match_count]])
    print(avg_distance)
    # 计算相似概率（距离越小，相似度越高）
    similarity_probability = max(0, 1 - avg_distance / 200)  # 使用200作为尺度因子

    # 显示匹配结果
    '''
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    return similarity_probability


# 读取两张图像
img1 = cv2.imread('img1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('img5.jpg', cv2.IMREAD_GRAYSCALE)

# 调用函数进行特征匹配
similarity = orb_feature_matching(img1, img2)
print(f"Similarity probability: {similarity}")
