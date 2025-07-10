import cv2
import numpy as np
import os


def check_exposure(input_image, under_threshold=15, over_threshold=180, print_info=False):
    """
    同时检测图像的欠曝光和过曝光区域，提高处理效率

    参数:
        input_image: 图像路径或图像数组
        under_threshold: 欠曝光阈值，低于此值的像素被视为欠曝光
        over_threshold: 过曝光阈值，高于此值的像素被视为过曝光
        print_info: 是否打印曝光信息

    返回:
        under_percentage: 欠曝光区域百分比
        over_percentage: 过曝光区域百分比
        under_binary_mask: 欠曝光区域的二值掩码
        over_binary_mask: 过曝光区域的二值掩码
        image: 原始图像
    """
    # 判断输入是路径还是图像数组
    if isinstance(input_image, str):
        # 如果是路径，读取图像
        image = cv2.imread(input_image)
        if image is None:
            raise ValueError(f"无法读取图像: {input_image}")
    else:
        # 如果已经是图像数组，直接使用
        image = input_image

    # 转换图像到Lab颜色空间 - 只转换一次
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, _, _ = cv2.split(lab_image)
    total_pixels = L.size

    # 检测欠曝光区域
    _, under_binary_mask = cv2.threshold(L, under_threshold, 255, cv2.THRESH_BINARY_INV)
    underexposed_pixels = np.sum(under_binary_mask > 0)
    under_percentage = (underexposed_pixels / total_pixels) * 100

    # 检测过曝光区域
    _, over_binary_mask = cv2.threshold(L, over_threshold, 255, cv2.THRESH_BINARY)
    overexposed_pixels = np.sum(over_binary_mask > 0)
    over_percentage = (overexposed_pixels / total_pixels) * 100

    return under_percentage, over_percentage, under_binary_mask, over_binary_mask, image


# 伽马校正函数
def adjust_gamma(image, gamma=2.0):
    # 建立查找表：将像素值（0-255）映射到经过伽马调整后的新值
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")

    # 应用查找表来调整图像
    return cv2.LUT(image, table)  # cv2.LUT 是 OpenCV 提供的函数，作用是根据查找表 table 将图像 image 的每个像素值转换为伽马校正后的新像素值。


def improved_MSRCR(img, scales=[10, 50, 150], k=0.02):
    # 将图像转换为浮点类型并归一化到 [0, 1]
    img_float = np.array(img, dtype="float32") / 255.0
    img_log = np.log1p(img_float)  # 对数变换
    result = np.zeros_like(img_log)
    img_light = np.zeros_like(img_log)
    r, c, deep = img_log.shape

    for z in range(deep):  # 遍历每个颜色通道
        for scale in scales:  # 遍历每个尺度
            kernel_size = scale * 4 + 1  # 高斯核大小
            sigma = scale  # 高斯核的标准差
            G_ratio = sigma ** 2 / (sigma ** 2 + k)  # 权重系数

            # 对当前通道应用高斯模糊
            img_smooth = cv2.GaussianBlur(img_log[:, :, z], (kernel_size, kernel_size), sigma)
            img_detail = img_log[:, :, z] - img_smooth  # 计算细节部分

            # 累加细节部分，并乘以权重系数
            result[:, :, z] += img_detail * G_ratio
            img_light[:, :, z] += img_smooth  # 累加光照分量

    # 合并细节和光照分量，取指数并归一化
    img_msrcr = np.exp(result + img_light) - 1
    img_msrcr = np.uint8(cv2.normalize(img_msrcr, None, 0, 255, cv2.NORM_MINMAX))
    return img_msrcr


# 主程序
def process_image(image_path='./inputs', over_threshold=220, under_threshold=15, output_dir='./outputs', gamma=1.8,
                  scales=[10, 50, 150], k=0.02, print_info=False):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 检查输入路径是文件还是目录
    if os.path.isdir(image_path):
        # 如果是目录，处理目录下所有支持的图像文件
        results = []
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        for filename in os.listdir(image_path):
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                file_path = os.path.join(image_path, filename)
                result = process_single_image(file_path, over_threshold, under_threshold, output_dir)
                results.append((file_path, *result))
        return results
    else:
        # 如果是单个文件，直接处理
        return process_single_image(image_path, over_threshold, under_threshold, output_dir)


def process_single_image(image_path, over_threshold=220, under_threshold=15, output_dir='./outputs',
                         gamma=1.8, scales=[10, 50, 150], k=0.02, print_info=False):
    """处理单个图像文件"""
    # 获取基本文件名
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_fixed{ext}")

    # 检测过曝和欠曝区域
    under_percentage, over_percentage, under_exposure_mask, over_exposure_mask, image = check_exposure(
        image_path, under_threshold, over_threshold, print_info=False)

    # 初始化处理后的图像为原始图像的副本
    processed_image = image.copy()

    # 处理图像
    if under_percentage - over_percentage > 0.1:
        # 对欠曝区域应用低光照增强
        processed_image = adjust_gamma(processed_image, gamma=1.1 + under_percentage / 100)
    elif over_percentage - under_percentage > 0.1:
        # 对过曝区域处理，带平滑过渡
        processed_image = adjust_gamma(processed_image, gamma=0.8 - over_percentage / 200)

    # 保存处理后的图像
    if over_percentage > 0.1 or under_percentage > 0.1:
        cv2.imwrite(output_path, processed_image)
        # print(f"处理后的图像已保存到: {output_path}")
        return output_path, over_percentage, under_percentage
    # else:
    #     print("图像没有过曝或欠曝区域，无需处理")
    #     return None, over_percentage, under_percentage


# ...existing code...

def process_webcam(frame, under_threshold=15, over_threshold=180, gamma=1.8, scales=[10, 50, 150], k=0.02):
    """
    使用笔记本摄像头作为输入，实时处理视频流

    参数:
        under_threshold: 欠曝光阈值
        over_threshold: 过曝光阈值
        gamma: 伽马校正参数
        scales: MSRCR算法的尺度参数
        k: MSRCR算法的参数
    """

    # 检测曝光情况
    under_percentage, over_percentage, under_exposure_mask, over_exposure_mask, _ = check_exposure(
        frame, under_threshold, over_threshold)

    # 处理图像
    processed_frame = frame.copy()

    if under_percentage - over_percentage > 0.1:
        # 对欠曝区域应用低光照增强
        processed_frame = adjust_gamma(processed_frame, gamma=1.1 + under_percentage / 100)
    elif over_percentage - under_percentage > 0.1:
        # 对过曝区域处理
        processed_frame = adjust_gamma(processed_frame, gamma=0.8 - over_percentage / 200)

    # 显示原始和处理后的图像
    info_text = f"欠曝: {under_percentage:.1f}% | 过曝: {over_percentage:.1f}%"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(processed_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 横向拼接两个图像
    display_image = np.hstack((frame, processed_frame))

    # 如果图像太大，缩小以适应屏幕
    if display_image.shape[1] > 1920:
        scale = 1920 / display_image.shape[1]
        display_image = cv2.resize(display_image, None, fx=scale, fy=scale)

    return processed_frame

    '''
        cv2.imshow('原始图像 | 处理后图像', display_image)

        # 检测按键，按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()
    '''

# 使用示例
if __name__ == "__main__":
    # 添加命令行参数解析
    import argparse

    parser = argparse.ArgumentParser(description='图像曝光修正工具')
    parser.add_argument('--webcam', action='store_true', help='使用摄像头实时处理')
    parser.add_argument('--input', default='./inputs', help='输入图像或目录路径')
    parser.add_argument('--output', default='./outputs', help='输出目录路径')
    parser.add_argument('--under', type=float, default=30, help='欠曝光阈值')
    parser.add_argument('--over', type=float, default=180, help='过曝光阈值')
    parser.add_argument('--gamma', type=float, default=1.8, help='伽马校正参数')

    args = parser.parse_args()

    if args.webcam:
        # 使用摄像头模式
        process_webcam(under_threshold=args.under, over_threshold=args.over, gamma=args.gamma)
    else:
        # 使用原来的文件处理模式
        process_image(args.input, output_dir=args.output,
                      under_threshold=args.under, over_threshold=args.over,
                      gamma=args.gamma, print_info=False)

# # 使用示例
# if __name__ == "__main__":
#     process_image("./inputs", output_dir="./outputs", under_threshold=30, over_threshold=180, gamma=1.8, scales = [10, 50, 150], k=0.02,print_info=False)


