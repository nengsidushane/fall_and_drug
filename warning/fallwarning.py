from warning.audiowarning import audiowarningfall
from warning.mailwarning import mailwarning
from warning.messagewarning import messagewarningfall, messagewarningmedicine
from warning.lightwarning import lightwarningfall
import cv2

def fallwarning(phone, to_email, photo):
    #lightwarningfall()
    audiowarningfall()
    #messagewarningmedicine(phone = phone)
    # 将 NumPy 数组编码为 JPEG 字节流
    _, img_encoded = cv2.imencode('.jpg', photo)
    img_bytes = img_encoded.tobytes()

    mailwarning(to_email = to_email, photo = img_bytes)
