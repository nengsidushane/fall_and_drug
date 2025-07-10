import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

def send_email(sms_content, to_email, smtp_server, smtp_port, smtp_username, smtp_password, photo_bytes):
    # 创建一个邮件对象
    msg = MIMEMultipart()
    msg['From'] = smtp_username
    msg['To'] = to_email
    msg['Subject'] = '慧眸守孤衾'

    # 添加短信内容到邮件正文
    body = MIMEText(sms_content, 'plain')
    msg.attach(body)

    # 直接使用字节流创建附件
    img_part = MIMEImage(photo_bytes, name='medicine_detection.jpg')
    img_part.add_header('Content-Disposition', 'attachment', filename='fall_detection.jpg')
    msg.attach(img_part)

    # 发送邮件
    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(smtp_username, smtp_password)
        server.sendmail(smtp_username, to_email, msg.as_string())
        print("邮件发送成功")
    except Exception as e:
        print(f"发送失败：{e}")
    finally:
        server.quit()

def mailwarning(to_email, photo):
    # 这里假设你已经收到了短信内容（sms_content）和指定的接收邮箱地址（to_email）
    # 你可以根据你的需求替换这些变量
    # to_email = "1094010353@qq.com"

    # 配置你的 SMTP 邮件服务器
    smtp_server = "smtp.qq.com"
    smtp_port = 465  # SMTP 端口，通常是 587 或 465
    smtp_username = "1094010353@qq.com"
    smtp_password = "ftftplprymvxbagc"

    sms_content = "【慧眸守孤衾】\n警告原因：检测到有人跌倒"
    sms_content = "【慧眸守孤衾】\n警告原因：检测到老人十分钟前就应该服用一袋复方板蓝根颗粒，但是并未按时服用，请及时督促服药"
    # 发送短信内容到指定邮箱
    send_email(sms_content, to_email, smtp_server, smtp_port, smtp_username, smtp_password, photo)
