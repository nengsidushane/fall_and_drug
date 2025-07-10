import cv2

# 设置摄像头参数（根据实际情况调整）
camera_index = 0           # 摄像头设备索引
total_width = 2560         # 双目合并后的总宽度
total_height = 720         # 双目合并后的高度
fps = 30                   # 帧率（根据摄像头实际帧率调整）

# 初始化摄像头
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, total_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, total_height)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 获取实际分辨率（验证设置是否生效）
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"摄像头分辨率: {actual_width}x{actual_height}")

# 计算左右画面分割位置
split_pos = actual_width // 2
left_size = (split_pos, actual_height)
right_size = (split_pos, actual_height)

# 创建视频编码器（MP4V格式）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 创建视频写入对象
left_video = cv2.VideoWriter('left_output11.mp4', fourcc, fps, left_size)
right_video = cv2.VideoWriter('right_output11.mp4', fourcc, fps, right_size)

print("开始录制，按Q停止...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧，退出...")
        break

    # 分割左右画面
    left_frame = frame[:, :split_pos]
    right_frame = frame[:, split_pos:]

    # 写入视频文件
    left_video.write(left_frame)
    right_video.write(right_frame)

    # 显示预览（可选）
    cv2.imshow('Left', left_frame)
    cv2.imshow('Right', right_frame)

    # 按Q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
left_video.release()
right_video.release()
cv2.destroyAllWindows()
print("录制完成，视频已保存为 left_output.mp4 和 right_output.mp4")