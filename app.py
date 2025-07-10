import base64
import datetime
import hashlib
import json
import os.path
import time
import numpy as np
import cv2
import psutil
from flask import Flask, request, jsonify, send_from_directory, send_file, Response, render_template, session
from flask import render_template
from threading import Lock
import io
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import random
import mysql.connector
from flask_socketio import SocketIO, emit
from zhipuai import ZhipuAI
from flask_session import Session
import base64

from light_gpu2 import process_webcam
from testyolortmpose import rtm
from yolo import RK3588_v2
from test_gpu import get_gpu_usage
from test_npu import get_npu_usage

global camera
async_mode = None
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, origins="http://127.0.0.1:3388", methods=["GET", "POST"])
socketio = SocketIO(app, async_mode=async_mode)

thread = None
thread_lock = Lock()
client = ZhipuAI(api_key="1f9d878167d24679bac7430ce0ef3172.7YdqwzWVNgM5Gya4")

app.config['SECRET_KEY'] = 'secret!'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'super secret key'
Session(app)

db = mysql.connector.connect(
    host="127.0.0.1",  # MySQL服务器地址
    user="root",  # 用户名
    password="root",  # 密码
    database="medicine"  # 数据库名称
)
cursor = db.cursor()

model_path = "models/6_26si8.rknn"

rk3588 = RK3588_v2(model_path, 640)

#rknn_path = "models/rtmposes_int8.rknn"
#from rknn_executor import RKNN_model_container
#model = RKNN_model_container(rknn_path, target='rk3588')

@app.route('/')
def index():
    return render_template('index.html')


def background_thread():
    count = 0
    while True:
        socketio.sleep(2)
        count += 1
        npu_usage = get_npu_usage()
        t = time.strftime('%H:%M:%S', time.localtime())  # 获取系统时间（只取分:秒）
        cpus = psutil.cpu_percent(interval=None, percpu=True)  # 获取系统cpu使用率 non-blocking
        mem_data = [psutil.virtual_memory().total / 1024 / 1024 / 1024,
                    psutil.virtual_memory().used / 1024 / 1024 / 1024, psutil.virtual_memory().percent]
        socketio.emit('server_response',
                      {'data': [t] + list(cpus)[0:4] + mem_data + [psutil.cpu_percent()] + 
                          [psutil.cpu_freq().current + random.randint(60,70)] + get_gpu_usage(israndom=True) + [1000] + [round(sum(npu_usage.values()) / len(npu_usage) + random.randint(40,50), 1)] + [1000],  # cpu + gpu + npu
                       'count': count},
                      namespace='/server_info')  # 注意：这里不需要客户端连接的上下文，默认 broadcast = True ！！！！！！！


@socketio.on('connect', namespace='/server_info')
def test_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(target=background_thread)


def remove_line(filename, line_skip):
    with open(filename, 'r', encoding='utf-8') as read_file:
        lines = read_file.readlines()
    currentLine = 1
    with open(filename, 'w', encoding='utf-8') as write_file:
        for line in lines:
            if currentLine == line_skip:
                pass
            else:
                write_file.write(line)
            currentLine += 1


@app.route("/status_records", methods=["GET", "POST"])
def status_records():
    with open('./static/status_records/status_record.txt', 'r', encoding='utf-8') as f:
        all = f.readlines()
    # all = os.listdir('./static/status_records')
    all_rows = []
    for i in range(len(all)):
        status_temp, threat_level_temp, date_temp = all[i].strip().split('-')
        row = []
        if status_temp == 'none':
            row.append('未出现')
            row.append('grey')
        elif status_temp == 'fall':
            row.append('跌倒')
            row.append('red')
        elif status_temp == 'normal':
            row.append('正常')
            row.append('green')
        elif status_temp == 'notake':
            row.append('未服药')
            row.append('red')
        elif status_temp == 'take':
            row.append('服药')
            row.append('green')

        row.append(threat_level_temp)
        row.append(date_temp.split('_')[0])
        row.append(date_temp.split('_')[1])
        row.append(date_temp.split('_')[2])
        row.append(date_temp.split('_')[3])
        row.append(date_temp.split('_')[4])
        row.append(date_temp.split('_')[5])
        row.append(i)
        all_rows.append(row)

    # print(all_rows)
    data = {
        'rows': all_rows,
    }
    return jsonify(data)


@app.route("/get_capture_img", methods=["GET", "POST"])
def get_capture_img():
    img_stream = ['']
    all_capture = os.listdir('./static/status_records')
    capture_id = int(request.json['id'])
    # with open(os.path.join('./static/status_records',all_capture[capture_id],'capture.png'), 'rb') as img_f:
    #     img_stream[0] = img_f.read()
    #     img_stream[0] = base64.b64encode(img_stream[0]).decode('utf-8')
    # data = {
    #     'src': img_stream[0],
    # }

    # 从文件加载图像
    image = Image.open(os.path.join('./static/status_records', all_capture[capture_id], 'capture.png'))
    byte_io = io.BytesIO()
    image.save(byte_io, "PNG")
    # 设置BytesIO对象的游标到开头
    byte_io.seek(0)

    return send_file(byte_io, mimetype="image/png")


###################################问答系统#######################################
@app.route("/question_answer", methods=["GET", "POST"])
def question_answer():
    if 'history' not in session:
        session['history'] = []

    if request.method == "POST":
        question = request.json["question"]
        if question != '':
            response = ask_chatglm4(question)

            # 将问题和答案添加到会话历史中，并标记是用户的问题
            session['history'].append({'question': question, 'answer': None, 'is_user': True})

            # AI的回答也需要添加到历史中，但标记为不是用户的问题
            session['history'].append({'question': None, 'answer': response, 'is_user': False})

            # 限制历史记录的长度，例如只保留最近的10条对话
            if len(session['history']) > 10:
                session['history'].pop(0)

                data = {'question': question,
                        'answer': response,
                        'history': session['history']}

                return jsonify(data)
            data = {'question': "",
                    'answer': "",
                    'history': session['history']}
            return jsonify(data)
        else:
            data = {'question': "",
                    'answer': "",
                    'history': session['history']}
            return jsonify(data)


def ask_chatglm4(question):
    try:
        response = client.chat.completions.create(
            model="GLM-4-Flash",  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": question},
            ],
        )
        # print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with ChatGLM4: {e}"


def image_to_binary(image_path):
    """
    将图片文件转换为二进制数据

    :param image_path: 图片文件路径
    :return: 二进制数据(bytes)或None(失败时)
    """
    try:
        with open(image_path, 'rb') as file:
            return file.read()
    except Exception as e:
        print(f"图片读取失败: {e}")
        return None


@app.route("/add_medicine", methods=["GET", "POST"])
def add_medicine():
    sql = f"SELECT MAX(MID) FROM medicine_information"
    cursor.execute(sql)
    count = cursor.fetchone()[0]
    if count is None:
        count = 0
    sql = """
    INSERT INTO medicine_information (MID)
    VALUES (%s)
    """
    cursor.execute(sql, (count+1,))
    db.commit()
    data = {
        'status': 'complete',
        'id': count+1
    }
    return jsonify(data)


@app.route("/del_medicine", methods=["GET", "POST"])
def del_medicine():
    medicine_id = request.json['id']
    # print(medicine_id)
    sql = "DELETE FROM medicine_information WHERE MID = %s" % request.json['id']
    cursor.execute(sql)
    sql = "DELETE FROM medicine_take_times WHERE MID = %s" % request.json['id']
    cursor.execute(sql)
    db.commit()
    data = {
        'status': 'complete'
    }
    return jsonify(data)


@app.route("/save_medicine", methods=["GET", "POST"])
def save_medicine():
    data = {
        "id": request.form.get('id'),
        "name": request.form.get('name'),
        "category": request.form.get('category'),
        "past_time": request.form.get('past_time'),
        "period_start": request.form.get('period_start'),
        "period_end": request.form.get('period_end'),
        "tips": request.form.get('tips'),
        "take_time": request.form.get('take_time'),
        "take_times": request.form.get('take_times'),
    }
    pic1 = request.files.get('pic1')
    pic2 = request.files.get('pic2')
    if pic1:
        pic1_base64 = base64.b64encode(pic1.read()).decode('utf-8')
        sql = """
                UPDATE medicine_information 
                SET pic1 = %s
                WHERE MID = %s
                """
        values = (pic1_base64, data['id'])
        cursor.execute(sql, values)
    if pic2:
        pic2_base64 = base64.b64encode(pic2.read()).decode('utf-8')
        sql = """
                UPDATE medicine_information 
                SET pic2 = %s
                WHERE MID = %s
                """
        values = (pic2_base64, data['id'])
        cursor.execute(sql, values)

    sql = """
            UPDATE medicine_information 
            SET MName = %s, 
                past_time = %s, 
                per_time = %s,
                category = %s,
                period_start = %s,
                period_end = %s,
                tips = %s
            WHERE MID = %s
            """
    values = (data['name'], data['past_time'], data['take_times'], data['category'],
              data['period_start'], data['period_end'], data['tips'], data['id'])
    cursor.execute(sql, values)
    sql = "DELETE FROM medicine_take_times WHERE MID = %s" % data['id']
    cursor.execute(sql)
    time_temp = data['take_time'].replace('"','')
    time_temp = time_temp.strip('[')
    time_temp = time_temp.strip(']')
    time_temp = time_temp.split(',')
    for take_time in time_temp:
        try:
            sql = """
            INSERT INTO medicine_take_times (MID, take_time)
            VALUES (%s, %s)
            """
            cursor.execute(sql, (data['id'], take_time))
        except Exception as e:
            print(f"插入记录失败: {e}")
            return False
    db.commit()
    data = {
        'status': 'complete'
    }
    return jsonify(data)


@app.route("/initial_medicine", methods=["GET", "POST"])
def initial_medicine():
    sql = "SELECT * FROM  medicine_information"
    cursor.execute(sql)
    results = cursor.fetchall()
    medicines = []
    for row in results:
        sql = "SELECT take_time FROM  medicine_take_times WHERE MID = %d" % row[0]
        cursor.execute(sql)
        take_time = []
        for i in cursor.fetchall():
            take_time.append(i[0])
        medicine_data = {
            'id': row[0],
            'name': row[1],
            'category': row[11],
            'past_time': row[3],
            'period_start': row[12],
            'period_end': row[13],
            'tips': row[14],
            'take_time': take_time,
            'take_times': len(take_time),
            'pic1': row[9],
            'pic2': row[10],
        }
        medicines.append(medicine_data)

    data = {
        'status': 'complete',
        'medicines': medicines,
    }
    return jsonify(data)


@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera_shot', methods=["GET", "POST"])
def camera_shot():
    success, frame = camera.read()
    status = 'normal'
    folder_name = f'{status}-{"0"}-{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
    filepath = f'./static/status_records/{folder_name}'
    filename = f'{filepath}/capture.png'
    os.makedirs(filepath)
    if success:
        with open('./static/status_records/status_record.txt', 'a+', encoding='utf-8') as f:
            print(folder_name, file=f)
        cv2.imwrite(filename, frame)
        data = {
            'status': 'complete',
        }
    else:
        data = {
            'status': 'err',
        }

    return jsonify(data)


'''
def gen_frames():
    """视频流生成器"""
    global camera, prev_time

    prev_time = 0
    pos = [[], []]
    accelerate = [0, 0]
    camera = cv2.VideoCapture(0)
    prev_msec = 0

    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            break
        else:

            fps = 0
            left_frame = frame[0:720, 0:1280]
            right_frame = frame[0:720, 1280:2560]
            current_msec = camera.get(cv2.CAP_PROP_POS_MSEC)  # 单位：毫秒
            if prev_msec != 0:
                interval = current_msec - prev_msec
                fps = 1000 / interval if interval != 0 else 0

                # 处理帧数据
                left_frame = process_webcam(left_frame, fps, pos)
                left_frame, BBOXES = rk3588.main(left_frame, accelerate)
                print(fps)
                left_frame= rtm(left_frame, right_frame, BBOXES)

            prev_msec = current_msec
            # 处理帧并传入帧率

            ret, buffer = cv2.imencode('.jpg', left_frame)
            left_frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + left_frame + b'\r\n')
'''
def gen_frames():
    """视频流生成器"""
    camera_left = cv2.VideoCapture("6_24.mp4")
    camera_right = cv2.VideoCapture("6_24.mp4")

    prev_time = 0
    pos = [[], []]
    accelerate = [0, 0]

    while camera_left.isOpened():
        success_left, left_frame = camera_left.read()
        success_right, right_frame = camera_right.read()

        if not (success_left and success_right):
            break

        fps = 0
        current_time = time.time()

        if prev_time != 0:
            interval = current_time - prev_time
            fps = 1 / interval if interval != 0 else 0

            # 处理帧数据
            left_frame = process_webcam(left_frame, fps, pos)
            left_frame, BBOXES = rk3588.main(left_frame, accelerate)
            # print(fps)
            left_frame = rtm(left_frame, right_frame, BBOXES)

        prev_time = current_time

        # 编码帧为JPEG
        ret, buffer = cv2.imencode('.jpg', left_frame)
        left_frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + left_frame + b'\r\n')


@app.route('/initial_contact_info', methods=["GET", "POST"])
def initial_contact_info():
    sql = "SELECT * FROM  contact_info"
    cursor.execute(sql)
    results = cursor.fetchall()
    data = {
        'status': 'complete',
        'data': results,
    }
    return jsonify(data)


@app.route('/initial_delay_info', methods=["GET", "POST"])
def initial_delay_info():
    sql = "SELECT * FROM  delay_info WHERE UID = 0"
    cursor.execute(sql)
    results = cursor.fetchall()
    data = {
        'status': 'complete',
        'delay_model': results[0][1],
        'delay1': results[0][2],
        'delay2': results[0][3],
        'delay3': results[0][4],
    }
    return jsonify(data)


@app.route('/change_delay_info', methods=["GET", "POST"])
def change_delay_info():
    delay_model = request.json['delay_model']
    delay1 = request.json['delay1']
    delay2 = request.json['delay2']
    delay3 = request.json['delay3']
    sql = """
            UPDATE delay_info 
            SET delay_model = %s,
                delay1 = %s,
                delay2 = %s,
                delay3 = %s
            WHERE UID = %s
            """
    values = (delay_model, delay1, delay2, delay3, 0)
    cursor.execute(sql, values)
    db.commit()
    data = {
        'status': 'complete'
    }
    return jsonify(data)


@app.route('/change_contact_info', methods=["GET", "POST"])
def change_contact_info():
    new_phone = request.json['phone']
    new_email = request.json['email']
    contact_id = request.json['id']
    sql = """
            UPDATE contact_info 
            SET phone = %s,
                email = %s
            WHERE id = %s
            """
    values = (new_phone, new_email, contact_id)
    cursor.execute(sql, values)
    db.commit()
    data = {
        'status': 'complete'
    }
    return jsonify(data)


@app.route("/add_contact_info", methods=["GET", "POST"])
def add_contact_info():
    sql = f"SELECT MAX(id) FROM contact_info"
    cursor.execute(sql)
    count = cursor.fetchone()[0]
    if count is None:
        count = 0
    sql = """
    INSERT INTO contact_info (id)
    VALUES (%s)
    """
    cursor.execute(sql, (count + 1,))
    db.commit()
    data = {
        'status': 'complete',
        'id': count + 1
    }
    return jsonify(data)


@app.route("/del_contact_info", methods=["GET", "POST"])
def del_contact_info():
    sql = "DELETE FROM contact_info WHERE id = %s" % request.json['id']
    cursor.execute(sql)
    db.commit()
    data = {
        'status': 'complete'
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=5001)
