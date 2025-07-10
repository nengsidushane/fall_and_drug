import urllib


def md5(str):
    import hashlib
    m = hashlib.md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()


statusStr = {
    '0': '短信发送成功',
    '-1': '参数不全',
    '-2': '服务器空间不支持,请确认支持curl或者fsocket,联系您的空间商解决或者更换空间',
    '30': '密码错误',
    '40': '账号不存在',
    '41': '余额不足',
    '42': '账户已过期',
    '43': 'IP地址限制',
    '50': '内容含有敏感词'
}


def messagewarningfall(phone):
    smsapi = "http://api.smsbao.com/"
    # 短信平台账号
    user = 'tomori'
    # 短信平台密码
    password = md5('why.nx0314')
    content = "【慧眸守孤衾】\n警告原因：检测到有人跌倒"
    data = urllib.parse.urlencode({
        'u': user,
        'p': password,
        'm': phone,
        'c': content
    })
    print(phone)
    send_url = smsapi + 'sms?' + data
    response = urllib.request.urlopen(send_url)
    the_page = response.read().decode('utf-8')
    print(statusStr[the_page])

def messagewarningmedicine(phone):
    smsapi = "http://api.smsbao.com/"
    # 短信平台账号
    user = 'tomori'
    # 短信平台密码
    password = md5('why.nx0314')
    content = "【慧眸守孤衾】\n警告原因：检测到老人十分钟前就应该服用一袋复方板蓝根颗粒，但是并未按时服用，请及时督促服药"
    data = urllib.parse.urlencode({
        'u': user,
        'p': password,
        'm': phone,
        'c': content
    })
    print(phone)
    send_url = smsapi + 'sms?' + data
    response = urllib.request.urlopen(send_url)
    the_page = response.read().decode('utf-8')
    print(statusStr[the_page])