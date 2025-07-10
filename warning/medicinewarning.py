import threading
from datetime import datetime, timedelta

from warning.audiowarning import audiowarningmedicine
from warning.lightwarning import lightwarningmedicine
from warning.messagewarning import messagewarningmedicine


def medicinewarning(target_time_str, phone):
    """
    在目标时间到达后，分别延迟1/5/10分钟执行f/g/h函数
    参数格式："HH:MM"（例如"17:56"）
    """
    try:
        now = datetime.now()
        # 解析输入时间为时间对象（只包含时/分）
        input_time = datetime.strptime(target_time_str, "%H:%M").time()
        # 将输入时间与当前日期组合
        target_time = datetime.combine(now.date(), input_time)

        print(now)

        # 如果目标时间已经过去，则设置为明天同一时间
        if target_time < now:
            target_time += timedelta(days=1)

        # 计算延迟时间（秒）
        delay = (target_time - now).total_seconds()

        # 定义定时器封装函数（修正参数传递问题）
        def create_timer(func, args=None, kwargs=None, minutes=0):
            args = args or ()
            kwargs = kwargs or {}
            timer_delay = delay + minutes * 60
            threading.Timer(timer_delay, func, args=args, kwargs=kwargs).start()

        # 创建三个定时任务（传递函数引用而非调用结果）
        create_timer(lightwarningmedicine, minutes=1)
        create_timer(audiowarningmedicine, minutes=5)
        create_timer(messagewarningmedicine, kwargs={'phone': phone}, minutes=10)

    except ValueError:
        print("时间格式错误，请使用'HH:MM'格式（例如'17:56'）")


# 使用示例（假设当前日期是2023-10-23）
medicinewarning("15:35", "13356953349")