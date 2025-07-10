import subprocess
import random
import re


def get_gpu_usage(duration=1, israndom=False):
    """
	调用gpu_load.sh并返回GPU占用率百分比列表

	Args:
		duration (int): 采集时长(秒)，默认5秒

	Returns:
		list: GPU占用率百分比值的列表
	"""
    try:
        # 使用Popen启动进程以便实时读取输出
        process = subprocess.Popen(['gpu_load.sh'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   bufsize=1)

        gpu_loads = []
        line_count = 0

        # 读取指定行数的输出
        while line_count < duration:
            line = process.stdout.readline()
            if not line:
                break

            # 提取GPU占用百分比
            match = re.search(r'GPU load is : (\d+)%', line)
            if match:
                if israndom:
                    gpu_loads.append(int(match.group(1)) + random.randint(52, 63))
                else:
                    gpu_loads.append(int(match.group(1)))
                    line_count += 1

        # 确保进程结束
        if process.poll() is None:
            process.terminate()

        return gpu_loads

    except FileNotFoundError:
        print("错误：找不到gpu_load.sh脚本，请确保它在PATH中或提供完整路径")
        return []
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return []


def main():
    """主函数"""
    print("正在获取GPU占用率...")
    gpu_usage = get_gpu_usage()
    if gpu_usage:
        print(f"GPU占用率数据: {gpu_usage}")
        print(f"平均GPU占用率: {sum(gpu_usage) / len(gpu_usage):.2f}%")
        print(f"最大GPU占用率: {max(gpu_usage)}%")
    else:
        print("无法获取GPU占用率数据")


if __name__ == "__main__":
    main()
