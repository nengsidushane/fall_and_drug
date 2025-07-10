import subprocess
import os

#def lightwarningfall():
    #os.system("echo heartbeat | sudo tee /sys/class/leds/green_led/trigger")

def lightwarningfall():
    try:
        subprocess.run(
            ["sudo", "bash", "-c", "echo heartbeat > /sys/class/leds/green_led/trigger"],
            check=True,
            stderr=subprocess.PIPE
        )
        print("绿灯已设置为心跳闪烁模式")
    except subprocess.CalledProcessError as e:
        print(f"执行失败：{e.stderr.decode()}")
    try:
        subprocess.run(
            ["sudo", "bash", "-c", "echo none > /sys/class/leds/blue_led/trigger"],
            check=True,
            stderr=subprocess.PIPE
        )
        #print("绿灯已设置为心跳闪烁模式")
    except subprocess.CalledProcessError as e:
        print(f"执行失败：{e.stderr.decode()}")

def lightwarningmedicine():
    try:
        subprocess.run(
            ["sudo", "bash", "-c", "echo heartbeat > /sys/class/leds/blue_led/trigger"],
            check=True,
            stderr=subprocess.PIPE
        )
        print("蓝灯已设置为心跳闪烁模式")
    except subprocess.CalledProcessError as e:
        print(f"执行失败：{e.stderr.decode()}")
    try:
        subprocess.run(
            ["sudo", "bash", "-c", "echo none > /sys/class/leds/green_led/trigger"],
            check=True,
            stderr=subprocess.PIPE
        )
        #print("绿灯已设置为心跳闪烁模式")
    except subprocess.CalledProcessError as e:
        print(f"执行失败：{e.stderr.decode()}")

if __name__ == "__main__":
    lightwarningfall()