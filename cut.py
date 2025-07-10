import cv2

def medicine_cut(image, left_x, left_y, right_x, right_y):
    '''
    通过传入照片左上角和右下角坐标，裁剪照片
        :param image: 传入的照片
        :param left_x: 左上角x坐标
        :param left_y: 左上角y坐标
        :param right_x: 右下角x坐标
        :param right_y: 右下角y坐标
        :return: 裁剪后的照片（result）
    '''
    # 确保坐标在有效范围内
    height, width = image.shape[:2]
    left_x = max(0, left_x)
    left_y = max(0, left_y)
    right_x = min(width, right_x)
    right_y = min(height, right_y)
    
    # 裁剪图像
    result = image[left_y:right_y, left_x:right_x]
    print("cut")
    cv2.imwrite('cropped_image.png', result)  # 保存裁剪后的图像
    #return result

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 读取图像
    image_path = '4.png'  # 替换为你的图像路径
    image = cv2.imread(image_path)
    
    # 定义裁剪区域的坐标
    left_x, left_y = 100, 100  # 左上角坐标
    right_x, right_y = 400, 400  # 右下角坐标
    
    # 调用裁剪函数
    cropped_image = cut(image, left_x, left_y, right_x, right_y)
    cv2.imwrite('cropped_image.png', cropped_image)  # 保存裁剪后的图像
    