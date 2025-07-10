import queue
import math

q_pos = queue.Queue(maxsize=5)
q_speed = queue.Queue(maxsize=2)
q_angle = queue.Queue(maxsize=5)
ref_high = 175
standard_accelerate = 0
standard_angle = 0


# pos = [[nose_x, nose_y, nose_z], [feet_x, feet_y, feet_z]]
def xyz_accelerate(pos):
    # 解包输入数据
    nose = pos[0]
    feet = pos[1]

    # 检查是否为全零坐标 ([0,0,0]判断)
    is_zero_nose = all(abs(v) < 1e-6 for v in nose)
    is_zero_feet = all(abs(v) < 1e-6 for v in feet)

    if is_zero_nose and is_zero_feet:
        return None, None  # 无效输入直接返回

    sigmoid_accelerate = None
    sigmoid_angle = None
    nose = pos[0]
    feet = pos[1]
    act_dis = 0
    for i in range(3):
        act_dis += math.pow(nose[i] - feet[i], 2)
    act_dis = act_dis ** 0.5
    act_angle = math.acos((nose[2] - feet[2]) / act_dis)
    if not q_angle.full():
        q_angle.put(act_angle)
    else:
        angle = q_angle.get()
        angle_accelerate = abs(angle - act_angle)
        sigmoid_angle = 1 / (1 + math.exp(standard_angle - angle_accelerate))

    if not q_pos.full():
        q_pos.put(nose)
    else:
        pos_g = q_pos.get()
        speed = math.sqrt(
            math.pow(pos_g[0] - nose[0], 2) + math.pow(pos_g[1] - nose[1], 2) + math.pow(pos_g[2] - nose[2], 2)) / 5
        q_pos.put(nose)

        if not q_speed.full():
            q_speed.put(speed)
        else:
            accelerate = abs(speed - q_speed.get())
            q_speed.put(speed)
            act_accelerate = accelerate * ref_high / act_dis
            sigmoid_accelerate = 1 / ( 1 + math.exp(standard_accelerate - act_accelerate))
        # 最终返回时确保包装成列表
    return [
            sigmoid_accelerate if sigmoid_accelerate is not None else 0,#速度置信度
            sigmoid_angle if sigmoid_angle is not None else 0#角度置信度
        ]
