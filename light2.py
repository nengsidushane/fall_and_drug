import cv2
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# 预计算伽马校正查找表
def create_gamma_table(gamma):
    """预计算并返回伽马校正查找表"""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return table

# 全局缓存常用的伽马查找表 - 添加更多预计算值以减少运行时创建
GAMMA_TABLES = {}
for gamma in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
    GAMMA_TABLES[gamma] = create_gamma_table(gamma)

def fast_check_exposure(image, under_threshold=15, over_threshold=180):
    """
    极速检测图像的欠曝光和过曝光区域
    """
    # 降采样以加速计算
    h, w = image.shape[:2]
    if h > 1280 or w > 720:
        # 降低分辨率到320x240或更小
        scale_factor = min(720/w, 1280/h)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        small_img = cv2.resize(image, (new_w, new_h))
    else:
        small_img = image
    
    # 转换为灰度图 - 使用更快的方法
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size
    
    # 使用直方图计算欠曝和过曝比例
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    under_pixels = np.sum(hist[:under_threshold])
    over_pixels = np.sum(hist[over_threshold:])
    
    under_percentage = (under_pixels / total_pixels) * 100
    over_percentage = (over_pixels / total_pixels) * 100
    
    return under_percentage, over_percentage

def check_exposure(input_image, under_threshold=15, over_threshold=180, print_info=False):
    """
    同时检测图像的欠曝光和过曝光区域，提高处理效率
    """
    # 判断输入是路径还是图像数组
    if isinstance(input_image, str):
        image = cv2.imread(input_image)
        if image is None:
            raise ValueError(f"无法读取图像: {input_image}")
    else:
        image = input_image
    
    # 使用快速曝光检测
    under_percentage, over_percentage = fast_check_exposure(image, under_threshold, over_threshold)
    
    # 创建掩码仅当需要时
    under_binary_mask = None
    over_binary_mask = None
    
    if under_percentage > 0.1 or over_percentage > 0.1:
        # 只在需要掩码时才计算掩码
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if under_percentage > 0.1:
            under_binary_mask = (gray < under_threshold).astype(np.uint8) * 255
        if over_percentage > 0.1:
            over_binary_mask = (gray > over_threshold).astype(np.uint8) * 255
    
    if print_info:
        print(f"欠曝光区域: {under_percentage:.2f}%, 过曝光区域: {over_percentage:.2f}%")
    
    return under_percentage, over_percentage, under_binary_mask, over_binary_mask, image

# 优化的伽马校正函数
def adjust_gamma(image, gamma=2.0):
    """使用缓存的查找表应用伽马校正"""
    # 量化gamma值到0.1的精度以提高查找表命中率
    gamma = round(gamma * 10) / 10
    
    # 使用预计算表
    if gamma in GAMMA_TABLES:
        table = GAMMA_TABLES[gamma]
    else:
        # 仅在必要时创建新表
        table = create_gamma_table(gamma)
        # 缓存新值以加速后续处理
        if len(GAMMA_TABLES) < 50:  # 限制缓存大小
            GAMMA_TABLES[gamma] = table
    
    # 应用查找表
    return cv2.LUT(image, table)

# 高度优化的MSRCR算法
def fast_MSRCR(img, scale=15, k=0.02):
    """
    单尺度视网膜色彩恢复算法 - 极速版
    """
    # 降低分辨率以加速处理
    h, w = img.shape[:2]
    max_size = 400
    if max(h, w) > max_size:
        scale_factor = max_size / max(h, w)
        small_img = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)))
    else:
        small_img = img
    
    # 转换为浮点并归一化
    img_float = small_img.astype(np.float32) / 255.0
    
    # 一次创建所有通道的对数图像
    img_log = np.log1p(img_float)
    kernel_size = scale * 2 + 1
    
    # 确保核大小为奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    sigma = scale / 2
    G_ratio = sigma**2 / (sigma**2 + k)
    
    # 为每个通道应用高斯模糊并处理
    result = np.zeros_like(img_float)
    
    for z in range(3):
        img_smooth = cv2.GaussianBlur(img_log[:, :, z], (kernel_size, kernel_size), sigma)
        img_detail = img_log[:, :, z] - img_smooth
        result[:, :, z] = np.exp(img_smooth + G_ratio * img_detail) - 1
    
    # 裁剪并转换回 uint8
    result = np.clip(result, 0, 1) * 255
    result = result.astype(np.uint8)
    
    # 如果之前缩小了图像，恢复原始尺寸
    if max(h, w) > max_size:
        result = cv2.resize(result, (w, h))
    
    return result

# 主程序
def process_image(image_path='./inputs', over_threshold=220, under_threshold=15, output_dir='./outputs',gamma=1.8, scales=[15], k=0.02, print_info=False):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入路径是文件还是目录
    if os.path.isdir(image_path):
        # 如果是目录，处理目录下所有支持的图像文件
        results = []
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # 使用线程池并行处理多个图像
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            futures = []
            
            for filename in os.listdir(image_path):
                if any(filename.lower().endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(image_path, filename)
                    future = executor.submit(process_single_image, file_path, over_threshold, under_threshold, 
                                           output_dir, gamma, scales, k, print_info)
                    futures.append((file_path, future))
            
            # 收集结果
            for file_path, future in futures:
                result = future.result()
                results.append((file_path, *result))
        
        return results
    else:
        # 如果是单个文件，直接处理
        return process_single_image(image_path, over_threshold, under_threshold, output_dir, 
                                   gamma, scales, k, print_info)

def process_single_image(image_path, over_threshold=320, under_threshold=30, output_dir='./outputs', gamma=1.8, scales=[15], k=0.02, print_info=False):
    """处理单个图像文件"""
    # 获取基本文件名
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_fixed{ext}")

    # 检测过曝和欠曝区域
    under_percentage, over_percentage, under_exposure_mask, over_exposure_mask, image = check_exposure(
        image_path, under_threshold, over_threshold, print_info=print_info)

    # 仅在需要时进行处理
    if under_percentage > 0.1 or over_percentage > 0.1:
        processed_image = image.copy()
        
        if under_percentage - over_percentage > 0.1:
            # 对欠曝区域应用低光照增强
            gamma_value = 1.1 + under_percentage/100
            processed_image = adjust_gamma(processed_image, gamma=gamma_value)
        elif over_percentage - under_percentage > 0.1:
            # 对过曝区域处理
            gamma_value = max(0.5, 0.8 - over_percentage/200)
            processed_image = adjust_gamma(processed_image, gamma=gamma_value)
        
        cv2.imwrite(output_path, processed_image)
        return output_path, over_percentage, under_percentage
    
    return None, over_percentage, under_percentage

class ImageProcessor:
    """图像处理器类，管理处理队列和线程"""
    def __init__(self):
        self.processing_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = False
        self.processor_thread = None
        
    def start(self):
        """启动处理线程"""
        if not self.running:
            self.running = True
            self.processor_thread = threading.Thread(target=self._process_queue)
            self.processor_thread.daemon = True
            self.processor_thread.start()
    
    def stop(self):
        """停止处理线程"""
        self.running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=1.0)
    
    def _process_queue(self):
        """处理队列中的图像"""
        while self.running:
            try:
                task = self.processing_queue.get(block=True, timeout=0.1)
                if task:
                    frame, under_threshold, over_threshold, gamma, process_msrcr, scales, k = task
                    result = self._process_frame(frame, under_threshold, over_threshold, gamma, process_msrcr, scales, k)
                    self.result_queue.put(result)
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理错误: {e}")
                continue
    
    def _process_frame(self, frame, under_threshold, over_threshold, gamma, process_msrcr, scales, k):
        """处理单个帧"""
        # 检测曝光情况
        under_percentage, over_percentage = fast_check_exposure(frame, under_threshold, over_threshold)
        
        # 处理图像
        processed_frame = frame.copy()
        apply_msrcr = False
        
        if under_percentage - over_percentage > 0.1:
            # 对欠曝区域应用低光照增强
            gamma_value = max(1.0, min(2.0, 1.1 + under_percentage/100))
            processed_frame = adjust_gamma(processed_frame, gamma=gamma_value)
        elif over_percentage - under_percentage > 0.1:
            # 对过曝区域处理
            gamma_value = max(0.5, min(1.0, 0.9 - over_percentage/200))
            processed_frame = adjust_gamma(processed_frame, gamma=gamma_value)
            
            # 仅在明显过曝时考虑使用MSRCR算法
            apply_msrcr = over_percentage > 15 and process_msrcr
        
        # 应用MSRCR算法
        if apply_msrcr:
            processed_frame = fast_MSRCR(processed_frame, scale=scales[0], k=k)
        
        return processed_frame, under_percentage, over_percentage
    
    def submit_frame(self, frame, under_threshold, over_threshold, gamma, process_msrcr, scales, k):
        """提交帧进行处理，非阻塞"""
        try:
            self.processing_queue.put_nowait((frame, under_threshold, over_threshold, gamma, process_msrcr, scales, k))
            return True
        except queue.Full:
            return False
    
    def get_result(self):
        """获取处理结果，非阻塞"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

class FrameRateController:
    """帧率控制器，用于平滑参数变化"""
    def __init__(self, smooth_factor=0.8):
        self.last_time = time.time()
        self.fps = 0
        self.smooth_factor = smooth_factor
        self.last_gamma = 1.0
        self.target_gamma = 1.0
        
    def update(self):
        """更新帧率"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if dt > 0:
            instantaneous_fps = 1.0 / dt
            self.fps = self.fps * self.smooth_factor + instantaneous_fps * (1 - self.smooth_factor)
        
        return self.fps
        
    def smooth_gamma(self, target):
        """平滑gamma参数变化"""
        self.target_gamma = target
        self.last_gamma = self.last_gamma * self.smooth_factor + target * (1 - self.smooth_factor)
        return self.last_gamma

def create_ui_overlay(base_height=50, width=1280, font_path='C:/Windows/Fonts/simhei.ttf'):
    """预创建UI叠加层"""
    title_img = np.zeros((base_height, width, 3), dtype=np.uint8)
    try:
        font_large = ImageFont.truetype(font_path, 24)
        img_pil = Image.fromarray(title_img)
        draw = ImageDraw.Draw(img_pil)
        text = '原始图像（左）| 处理后图像（右）'
        text_width = draw.textlength(text, font=font_large)
        center_x = width//2 - text_width//2
        draw.text((center_x, 10), text, font=font_large, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"创建UI叠加层错误: {e}")
        return title_img

def process_webcam(frame, fps, under_threshold=30, over_threshold=220, gamma=1.8, scales=[15, 80], k=0.02):
    """
    使用笔记本摄像头作为输入，实时处理视频流 (优化版)
    """
    
    # 预加载字体和设置常量
    global processed_frame_with_text
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # 文泉驿微米黑
    try:
        font_small = ImageFont.truetype(font_path, 20)
        font_large = ImageFont.truetype(font_path, 24)
    except IOError:
        print("警告: 无法加载指定字体，将使用默认字体")
        font_small = ImageFont.load_default()
        font_large = ImageFont.load_default()
    
    # 帧率计算变量
    prev_time = time.time()
    fps_alpha = 0.1  # 平滑因子
    
    # 处理间隔控制
    process_every_n_frames = 1  # 每N帧处理一次
    frame_count = 0
    last_processed_frame = None
    last_display_image = None
    
    # 缓存处理结果
    last_under_percentage = 0
    last_over_percentage = 0

    # 计算当前帧率
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    # 平滑帧率显示
    if dt > 0:
        current_fps = 1.0 / dt
        fps = fps * (1 - fps_alpha) + current_fps * fps_alpha

    # 处理图像
    processed_frame = frame.copy()
    
    # 移除文本显示相关代码
    processed_frame_with_text = processed_frame
    
    frame_count += 1

    return processed_frame_with_text

# # 使用示例
# if __name__ == "__main__":
#     process_image("./inputs", output_dir="./outputs", under_threshold=30, over_threshold=180, gamma=1.8, scales = [10, 50, 150], k=0.02,print_info=False)


