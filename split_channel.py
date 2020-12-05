import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image
import numpy as np
import os

# img=Image.open(r'G:\xqy\faster_rcnn\FPCcrease_DATASET\JPEGImages\fpccrease_0.jpg')
# r,g,b,a=img.split()   #分离三通道
# pic=Image.merge('RGB',(r,g,b)) #合并三通道
# pic_np = np.array(pic)
# print(pic_np.shape)

if __name__ == '__main__':
        # path = r'C:\Users\Administrator\Desktop\fpc'   #运行程序前，记得修改主文件夹路径！
        path = r"G:\xqy\faster_rcnn\FPCcrease_DATASET\JPEGImages"
        reshape_path = os.path.join(path, "reshape")
        old_names = os.listdir(reshape_path)
        # print(old_names)
        # print(os.listdir(path))
        for old_name in old_names:
                image = Image.open(os.path.join(reshape_path, old_name))
                image_np = np.array(image)
                shape_original = image_np.shape
                r, g, b, a = image.split()
                pic = Image.merge('RGB', (r, g, b))  # 合并三通道
                pic_np = np.array(pic)
                shape_trans = pic_np.shape
                pic.save(os.path.join(path, "saved", old_name))
                print("%s successfully transform from %s to %s" %(old_name, (shape_original), str(shape_trans)))

