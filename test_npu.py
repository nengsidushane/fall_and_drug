import subprocess
import re

def get_npu_usage():
    """
    调用sudo cat /sys/kernel/debug/rknpu/load并返回NPU各核心占用率
    
    Returns:
        dict: 包含各核心占用率的字典，格式为 {'Core0': 值, 'Core1': 值, 'Core2': 值}
        如果出错则返回None
    """
    try:
        # 执行命令并获取输出
        result = subprocess.run(['sudo', 'cat', '/sys/kernel/debug/rknpu/load'], 
                               capture_output=True, 
                               text=True, 
                               check=True)
        
        # 提取各核心占用率
        output = result.stdout.strip()
        # 使用正则表达式匹配所有形如"Core0:  0%"的模式
        matches = re.findall(r'Core(\d+):\s+(\d+)%', output)
        
        if not matches:
            print("无法从输出中解析NPU负载数据")
            return None
            
        # 将结果存入字典
        npu_loads = {}
        for core_num, load_percent in matches:
            npu_loads[f'Core{core_num}'] = int(load_percent)
            
        return npu_loads
        
    except subprocess.CalledProcessError:
        print("执行sudo命令失败，请确保有足够权限")
        return None
    except FileNotFoundError:
        print("文件不存在：/sys/kernel/debug/rknpu/load")
        return None
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None

def main():
    """主函数"""
    print("正在获取NPU占用率...")
    npu_usage = get_npu_usage()
    
    if npu_usage:
        print("NPU占用率数据:")
        for core, load in npu_usage.items():
            print(f"{core}: {load}%")
        
        # 计算平均负载
        if npu_usage:
            avg_load = sum(npu_usage.values()) / len(npu_usage)
            print(f"平均NPU占用率: {avg_load:.2f}%")
    else:
        print("无法获取NPU占用率数据")

if __name__ == "__main__":
    main()