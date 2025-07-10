import cv2
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image
import time
import argparse # Moved import here
from concurrent.futures import ThreadPoolExecutor
#运行命令：python light_gpu.py --webcam 0
# --- 全局设置与辅助函数 ---

# 预计算伽马校正查找表
def create_gamma_table(gamma):
    """预计算并返回伽马校正查找表"""
    invGamma = 1.0 / max(gamma, 0.01) # Avoid division by zero or negative gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return table

# 全局缓存常用的伽马查找表
GAMMA_TABLES = {}
for gamma_val in np.round(np.arange(0.4, 2.6, 0.1), 1): # Precompute a useful range
    GAMMA_TABLES[gamma_val] = create_gamma_table(gamma_val)

# 优化的伽马校正函数 (兼容 UMat 和 np.ndarray)
def adjust_gamma(image, gamma=1.0):
    """
    使用缓存的查找表应用伽马校正。
    接受 cv2.UMat 或 np.ndarray 作为输入。
    """
    # 量化 gamma 值以匹配缓存的键
    gamma = round(max(0.1, gamma) * 10) / 10 # Ensure gamma is positive and round
    gamma = min(gamma, 2.5) # Clamp to precomputed range max if needed
    gamma = max(gamma, 0.4) # Clamp to precomputed range min if needed

    # 使用预计算表 (如果 gamma 在范围内)
    if gamma in GAMMA_TABLES:
        table = GAMMA_TABLES[gamma]
    else:
        # 对于超出预计算范围的值，动态计算（不缓存）
        # print(f"警告: Gamma 值 {gamma} 超出预计算范围 [0.4, 2.5]，动态计算。")
        table = create_gamma_table(gamma)
        # 不缓存动态计算的值，避免缓存膨胀

    # cv2.LUT 对 UMat 和 np.ndarray 都有效
    return cv2.LUT(image, table)

def fast_check_exposure(image, under_threshold=15, over_threshold=220):
    """
    更快的曝光检测算法 (在 CPU 上运行)。
    使用缩小的灰度图和 NumPy 进行快速计算。
    """
    h, w = image.shape[:2]
    # 进一步降低检查分辨率以提高速度
    target_pixels = 640 * 480
    current_pixels = h * w
    scale = 1.0
    if current_pixels > target_pixels:
        scale = np.sqrt(target_pixels / current_pixels)

    if scale < 0.95: # Only resize if significantly larger
        new_w, new_h = int(w * scale), int(h * scale)
        # 使用 INTER_NEAREST 以获得最快速度
        small_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        small_img = image

    # 转换为灰度图
    if len(small_img.shape) == 3 and small_img.shape[2] == 3:
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    elif len(small_img.shape) == 2:
        gray = small_img # Already grayscale
    else:
        # Fallback for unexpected shapes
        print("警告: fast_check_exposure 输入图像格式未知，尝试灰度转换。")
        try:
            gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        except:
            # If conversion fails, calculate on the first channel if possible
            if len(small_img.shape) >= 2:
                 gray = small_img[:,:,0] if len(small_img.shape) == 3 else small_img
            else: # Cannot process
                 return 0.0, 0.0


    # 使用 numpy 掩码操作计算百分比
    total_pixels = gray.size
    if total_pixels == 0: return 0.0, 0.0

    under_pixels = np.sum(gray < under_threshold)
    over_pixels = np.sum(gray > over_threshold)

    under_percentage = (under_pixels / total_pixels) * 100
    over_percentage = (over_pixels / total_pixels) * 100

    return under_percentage, over_percentage

# SSR 算法 (CPU 版本)
def fast_SSR(img, scale=15, k=0.02):
    """
    单尺度视网膜色彩恢复算法 - CPU 优化版
    """
    h, w = img.shape[:2]
    max_size = 300
    scale_factor = 1.0
    if max(h, w) > max_size:
        scale_factor = max_size / max(h, w)
        small_img = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)
    else:
        small_img = img

    img_float = small_img.astype(np.float32) / 255.0
    img_log = np.log1p(img_float) # 使用 log1p 避免 log(0)

    kernel_size = int(scale * 2 + 1)
    if kernel_size % 2 == 0: kernel_size += 1

    # 使用分离的 1D 高斯核可能更快，但 cv2.GaussianBlur 已优化
    img_smooth = cv2.GaussianBlur(img_log, (kernel_size, kernel_size), scale / 2.0) # Sigma 通常是 scale 的一部分

    # MSRCP/SSR 公式的一部分，这里简化处理
    # R(x,y) = log(I(x,y)) - log(I(x,y) * G(x,y))
    # 这里使用一种变体或简化形式，直接用高斯模糊结果
    # result_log = img_log - img_smooth # 基本 SSR 对数差分

    # 考虑一个更简单的对比度增强形式 (类似 DoG)
    # 或者使用论文中的公式（如果已知）
    # 假设是简单的对数域增强:
    # gain = 1.5 # 示例增益
    # result_log = gain * (img_log - img_smooth) + img_smooth # 增强细节并加回基底

    # --- 尝试恢复原始代码的意图 ---
    # 原始代码似乎混合了某种形式的 SSR 和可能的增益控制
    # sigma = scale / 2
    # G_ratio = sigma**2 / (sigma**2 + k) # 这个比例因子用途不明，可能是某种自适应
    # result_log = img_smooth + G_ratio * (img_log - img_smooth) # 原始公式?
    # result = np.expm1(result_log) # 使用 expm1 对应 log1p

    # --- 简化为标准对数差分SSR ---
    result_log = img_log - img_smooth
    result = np.expm1(result_log) # exp(x) - 1

    # 归一化和转换回 uint8
    # min_val, max_val = np.min(result), np.max(result)
    # if max_val > min_val:
    #    result = (result - min_val) / (max_val - min_val)
    # else:
    #    result = np.zeros_like(result) # Avoid division by zero

    # 直接裁剪可能更符合原始意图
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

    if scale_factor < 1.0:
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

    return result

# --- 文件处理 ---

def process_single_image(image_path, over_threshold=220, under_threshold=15, output_dir='./outputs', gamma=1.8, scales=[15], k=0.02, print_info=False):
    """
    处理单个图像文件 (尝试使用 UMat 进行伽马校正)。
    """
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_fixed{ext}")

    # 1. 读取图像 (CPU)
    image_cpu = cv2.imread(image_path)
    if image_cpu is None:
        print(f"警告: 无法读取图像 {image_path}")
        return None, 0, 0

    # 2. 检查曝光 (CPU)
    under_percentage, over_percentage = fast_check_exposure(
        image_cpu, under_threshold, over_threshold)

    if print_info:
         print(f"文件: {base_name}, 欠曝: {under_percentage:.2f}%, 过曝: {over_percentage:.2f}%")

    processed_image_to_save = None
    needs_processing = under_percentage > 0.1 or over_percentage > 0.1

    if needs_processing:
        use_gpu = False # 默认不使用 GPU，除非 UMat 成功
        try:
            # 3. 尝试将图像传输到 GPU (UMat)
            # 注意: 这一步有开销。只有当 GPU 处理速度显著快于 CPU + 开销时才有意义。
            image_umat = cv2.UMat(image_cpu)
            processed_image = image_umat # 在 UMat 上开始处理
            use_gpu = True # 标记成功使用了 UMat

            # 4. 在 UMat 上应用伽马校正 (如果 OpenCV 支持，将在 GPU 上运行)
            dynamic_gamma = gamma # 使用基础 gamma 或动态调整
            if under_percentage - over_percentage > 0.1:
                dynamic_gamma = min(2.5, 1.1 + under_percentage/100)
            elif over_percentage - under_percentage > 0.1:
                dynamic_gamma = max(0.4, 0.9 - over_percentage/200)

            processed_image = adjust_gamma(processed_image, gamma=dynamic_gamma)

            # 其他可以在 UMat 上执行的操作可以在这里添加...
            # 例如: resizing_umat = cv2.resize(processed_image, ...)

            # 5. 将结果从 GPU 传回 CPU 以便保存
            processed_image_to_save = processed_image.get()
            if print_info:
                 print(f"文件: {base_name} 使用 UMat (GPU) 处理完成。")

        except cv2.error as e:
            # 6. 如果 UMat 失败 (例如 OpenCV 不支持或配置错误)，回退到 CPU
            print(f"警告: UMat 处理失败 ({e})，回退到 CPU 处理 {base_name}")
            use_gpu = False
            processed_image = image_cpu.copy() # 在 CPU 副本上处理

            # 在 CPU 上应用伽马校正
            dynamic_gamma = gamma
            if under_percentage - over_percentage > 0.1:
                 dynamic_gamma = min(2.5, 1.1 + under_percentage/100)
            elif over_percentage - under_percentage > 0.1:
                 dynamic_gamma = max(0.4, 0.9 - over_percentage/200)

            processed_image = adjust_gamma(processed_image, gamma=dynamic_gamma) # adjust_gamma 也适用于 np.array

            processed_image_to_save = processed_image # 结果已经在 CPU 上

        # 7. 保存处理后的图像 (如果处理了)
        if processed_image_to_save is not None:
             try:
                 cv2.imwrite(output_path, processed_image_to_save)
                 return output_path, over_percentage, under_percentage
             except Exception as write_e:
                 print(f"错误: 无法写入输出文件 {output_path}: {write_e}")
                 return None, over_percentage, under_percentage
        else:
             # 理论上不应发生，因为 needs_processing 为 True
             print(f"信息: 文件 {base_name} 需要处理但未生成结果。")
             return None, over_percentage, under_percentage
    else:
        # 无需处理
        if print_info:
             print(f"信息: 文件 {base_name} 无需处理。")
        return None, over_percentage, under_percentage

def process_image_directory(image_dir='./inputs', over_threshold=220, under_threshold=15, output_dir='./outputs', gamma=1.8, scales=[15], k=0.02, print_info=False):
    """处理目录中的所有图像文件 (使用线程池)"""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    files_to_process = []

    for filename in os.listdir(image_dir):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            files_to_process.append(os.path.join(image_dir, filename))

    if not files_to_process:
        print(f"在目录 {image_dir} 中未找到支持的图像文件。")
        return []

    # 使用线程池并行处理
    # 限制 worker 数量以避免过多内存或 GPU 资源竞争
    max_workers = min(4, os.cpu_count() or 1)
    print(f"使用 {max_workers} 个线程处理 {len(files_to_process)} 个文件...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        futures = {executor.submit(process_single_image, file_path, over_threshold, under_threshold,
                                   output_dir, gamma, scales, k, print_info): file_path
                   for file_path in files_to_process}

        # 收集结果 (按完成顺序)
        for future in futures: # concurrent.futures.as_completed(futures): # Use as_completed for progress
            file_path = futures[future]
            try:
                result = future.result() # 获取结果 (output_path, over, under) or (None, over, under)
                if result is not None and result[0] is not None: # Check if processing occurred and saved
                    results.append((file_path, *result[1:])) # Store (input_path, over, under)
                    if print_info: print(f"完成: {os.path.basename(file_path)}")
                elif result is not None: # No processing needed or save failed
                     if print_info: print(f"跳过/失败: {os.path.basename(file_path)}")
                     results.append((file_path, *result[1:])) # Still record exposure info
            except Exception as exc:
                print(f"处理文件 {os.path.basename(file_path)} 时产生错误: {exc}")
                results.append((file_path, -1, -1)) # Indicate error

    print(f"目录处理完成，共处理 {len(results)} 个文件。")
    return results


# --- 摄像头实时处理 ---

def process_webcam(frame, fps, pos, under_threshold=15, over_threshold=180, base_gamma=1.8, scales=[15, 80], k=0.02):
    """
    处理单帧图像，用于实时摄像头流 (无状态, CPU 版本, 含内部缩放优化)。
    返回处理后的帧 (np.array) 和曝光百分比。
    """
    h, w = frame.shape[:2]
    processed_frame = frame # 默认返回原始帧

    # --- 优化: 在较小的副本上进行计算 ---
    scale_factor = 1.0
    target_pixels = 640 * 480 # 处理目标分辨率
    current_pixels = h * w
    if current_pixels > target_pixels * 1.1: # 仅当显著大于目标时才缩放
        scale_factor = np.sqrt(target_pixels / current_pixels)

    if scale_factor < 0.95:
        proc_w = int(w * scale_factor)
        proc_h = int(h * scale_factor)
        try:
            small_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        except Exception as resize_err:
            print(f"警告: 帧缩放失败: {resize_err}")
            small_frame = frame # 出错则使用原始帧
            scale_factor = 1.0 # 重置缩放因子
    else:
        small_frame = frame # 如果图像已经足够小，直接处理
        scale_factor = 1.0

    # 1. 检测曝光 (在小帧上进行以提高速度)
    under_percentage, over_percentage = fast_check_exposure(small_frame, under_threshold, over_threshold)

    # 2. 图像处理 (在小帧上进行)
    processed_small_frame = small_frame.copy() # 在副本上操作
    needs_processing = under_percentage > 0.1 or over_percentage > 0.1

    if needs_processing:
        dynamic_gamma = base_gamma
        if under_percentage - over_percentage > 5: # 阈值调整为5，更敏感
            # 欠曝: 增加 gamma (变亮)
            dynamic_gamma = min(2.5, 1.1 + under_percentage / 100)
        elif over_percentage - under_percentage > 5: # 阈值调整为5
            # 过曝: 减少 gamma (变暗)
            dynamic_gamma = max(0.4, 0.9 - over_percentage / 200)

        # 应用伽马校正 (CPU)
        processed_small_frame = adjust_gamma(processed_small_frame, gamma=dynamic_gamma)

        # --- SSR (可选, CPU密集型) ---
        # 仅在严重过曝时考虑，且可能需要进一步优化或跳帧执行
        # apply_ssr = over_percentage > 50 # 示例：仅在过曝 > 50% 时应用
        # if apply_ssr:
        #     try:
        #         # 注意: fast_SSR 可能很慢，会显著影响帧率
        #         processed_small_frame = fast_SSR(processed_small_frame, scale=scales[0], k=k)
        #     except Exception as ssr_err:
        #         print(f"警告: fast_SSR 执行失败: {ssr_err}")
        # --- SSR 结束 ---

    # 3. 如果之前缩小了，则放大回原始尺寸
    if scale_factor < 0.95:
        try:
            processed_frame = cv2.resize(processed_small_frame, (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception as resize_err:
            print(f"警告: 帧放大失败: {resize_err}")
            processed_frame = frame # 出错则返回原始帧
    elif needs_processing:
         processed_frame = processed_small_frame # 如果没缩小但处理了
    # else: processed_frame 保持为原始 frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_small = 0.6
    font_scale_large = 0.8
    green_color = (0, 255, 0)  # BGR
    
    # 添加帧率显示
    #cv2.putText(processed_frame, str(fps),
    #            (processed_frame.shape[1] - 1280, 60), font, font_scale_small, green_color, 2)

    
    #fps_text1 = "nose"+str(pos[0])
    #fps_text2 = "feet"+str(pos[1])
    # cv2.putText(processed_frame, fps_text1,
    #             (processed_frame.shape[1] - 1280, 30), font, font_scale_small, green_color, 2)
    # cv2.putText(processed_frame, fps_text2,
    #             (processed_frame.shape[1] - 1280, 60), font, font_scale_small, green_color, 2)

    # 4. 返回处理结果 (原始尺寸) 和曝光信息 (来自小帧检查)
    return processed_frame


# --- 主程序入口 ---

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='自适应光照算法 (尝试 UMat 加速文件处理)')
    parser.add_argument('--webcam', type=int, default=None, metavar='ID', help='使用指定ID的摄像头实时处理 (例如 0)')
    parser.add_argument('--input', default='./inputs', help='输入图像或目录路径 (文件处理模式)')
    parser.add_argument('--output', default='./outputs', help='输出目录路径 (文件处理模式)')
    parser.add_argument('--under', type=float, default=30, help='欠曝光阈值 (百分比下限)')
    parser.add_argument('--over', type=float, default=220, help='过曝光阈值 (百分比上限)')
    parser.add_argument('--gamma', type=float, default=1.0, help='基础伽马值 (动态调整的基准)')
    parser.add_argument('--scales', nargs='+', type=int, default=[15], help='SSR 尺度参数 (目前主要在CPU)')
    parser.add_argument('--k', type=float, default=0.02, help='SSR k 参数 (目前主要在CPU)')
    parser.add_argument('--skip', type=int, default=1, metavar='N', help='摄像头模式下每 N 帧处理 1 帧 (>=1)')
    parser.add_argument('--noinfo', action='store_true', help='不在文件处理模式下打印详细信息')

    args = parser.parse_args()

    # --- 摄像头模式 ---
    if args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {args.webcam}")
            exit()

        print(f"启动摄像头 {args.webcam}... 按 'q' 退出。")
        print(f"处理帧间隔: {args.skip}")

        # 状态变量
        prev_time = time.time()
        fps = 0.0
        fps_alpha = 0.1 # 平滑因子
        frame_count = 0
        last_processed_frame = None
        last_under = 0.0
        last_over = 0.0
        process_interval = max(1, args.skip) # 确保至少为 1

        # 字体加载 (尝试常见路径)
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', # 文泉驿微米黑 (Linux)
            'C:/Windows/Fonts/simhei.ttf', # SimHei (Windows)
            '/System/Library/Fonts/STHeiti Medium.ttc', # Heiti (macOS)
            None # 后备
        ]
        font_small = None
        for fp in font_paths:
            if fp and os.path.exists(fp):
                try:
                    font_small = ImageFont.truetype(fp, 18) # 稍小字体
                    print(f"加载字体: {fp}")
                    break
                except IOError:
                    continue
        if font_small is None:
            print("警告: 未找到支持的字体，使用默认字体。")
            font_small = ImageFont.load_default()


        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法从摄像头读取帧。")
                break

            current_time = time.time()
            time_diff = current_time - prev_time
            if time_diff > 1e-5: # 避免除零
                 instant_fps = 1.0 / time_diff
                 fps = fps * (1 - fps_alpha) + instant_fps * fps_alpha
            prev_time = current_time

            processed_frame = None
            under = last_under
            over = last_over
            print("FPS:{:.1f} under: {:.1f} over: {:.1f}".format(fps, under, over), end='\r')
            # 跳帧处理逻辑
            if frame_count % process_interval == 0:
                try:
                    processed_frame, under, over = process_frame_for_webcam(
                        frame,
                        under_threshold=args.under,
                        over_threshold=args.over,
                        base_gamma=args.gamma,
                        scales=args.scales,
                        k=args.k
                    )
                    last_processed_frame = processed_frame # 缓存结果
                    last_under = under
                    last_over = over
                except Exception as proc_err:
                    print(f"错误: 处理帧时发生错误: {proc_err}")
                    processed_frame = frame # 出错时显示原始帧
                    # 不更新 last_processed_frame
            else:
                # 使用上一帧的处理结果（如果存在）
                processed_frame = last_processed_frame if last_processed_frame is not None else frame

            # --- 显示与文本叠加 ---
            display_img = processed_frame.copy() # 在副本上绘制

            # 添加文本信息 (在 CPU 上使用 PIL)
            try:
                img_pil = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                info_text = f"欠:{under:.1f}% 过:{over:.1f}%"
                fps_text = f"FPS:{fps:.1f}"
                # 绘制背景以便阅读
                info_size = draw.textbbox((0,0), info_text, font=font_small)
                fps_size = draw.textbbox((0,0), fps_text, font=font_small)
                draw.rectangle((5, 5, 10 + info_size[2], 10 + info_size[3]), fill=(0, 0, 0, 128))
                draw.rectangle((display_img.shape[1] - fps_size[2] - 15, 5, display_img.shape[1] - 5, 10 + fps_size[3]), fill=(0, 0, 0, 128))
                # 绘制文本
                draw.text((8, 8), info_text, font=font_small, fill=(0, 255, 0)) # 绿色
                draw.text((display_img.shape[1] - fps_size[2] - 12, 8), fps_text, font=font_small, fill=(0, 255, 0)) # 绿色
                display_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as draw_err:
                print(f"警告: 绘制文本时出错: {draw_err}")
                # 如果绘图失败，至少显示图像

            cv2.imshow(f'Webcam {args.webcam} Processed (q to quit)', display_img)

            frame_count += 1

            # 退出条件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # elif key == ord('+') or key == ord('='): # 示例：增加处理频率
            #     process_interval = max(1, process_interval - 1)
            #     print(f"处理帧间隔: {process_interval}")
            # elif key == ord('-'): # 示例：降低处理频率
            #     process_interval += 1
            #     print(f"处理帧间隔: {process_interval}")


        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头模式已退出。")

    # --- 文件处理模式 ---
    else:
        if os.path.isdir(args.input):
            print(f"处理目录: {args.input}, 输出到: {args.output}")
            process_image_directory(args.input, output_dir=args.output,
                         under_threshold=args.under, over_threshold=args.over,
                         gamma=args.gamma, scales=args.scales, k=args.k,
                         print_info=not args.noinfo)
            print("目录处理完成。")
        elif os.path.isfile(args.input):
            print(f"处理文件: {args.input}, 输出到: {args.output}")
            os.makedirs(args.output, exist_ok=True) # 确保输出目录存在
            result_path, over, under = process_single_image(
                         args.input, output_dir=args.output,
                         under_threshold=args.under, over_threshold=args.over,
                         gamma=args.gamma, scales=args.scales, k=args.k,
                         print_info=not args.noinfo)
            if result_path:
                print(f"文件处理完成，结果保存在: {result_path}")
            elif not args.noinfo:
                 print(f"文件 {os.path.basename(args.input)} 无需处理或处理失败。")
        else:
            print(f"错误: 输入路径 '{args.input}' 不是有效的文件或目录。")

