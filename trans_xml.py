import os
from xml.etree.ElementTree import parse, Element


def trans_name(start_index=0):
    """
    批量修改文件名
    """
    for file in fileList:
        oldname = os.path.join(path, str(file))

        newname = os.path.join(path, ("tgkcrease_gan_"+str(start_index)+".jpg"))    # 修改newname

        os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
        print(oldname, '======>', newname)
        start_index = start_index + 1


def trans_xml():
    """
    批量修改xml文件
    """
    for file in fileList:
        filename = os.path.join(path, file)
        tree = parse(filename)
        root = tree.getroot()
        # sub1 = root.find("filename")
        # sub1.text = str(file).split(".")[0]+".jpg"
        sub2 = root.find("path")
        image_path = "G:\\xqy\\faster_rcnn\\DAGM_DATASET\\JPEGImages\\"
        sub2.text = image_path + file.split(".")[0] + ".jpg"
        tree.write(filename)
        print(filename+" changed.")


if __name__ == "__main__":
    # path = r"G:\xqy\faster_rcnn\DAGM_DATASET\Annotations"
    path = r"G:\xqy\SinGAN-master\Output\RandomSamples\tgkcrease_11_crop_14\gen_start_scale=3" # 要修改的文件路径
    fileList = os.listdir(path)  # test有1054个,train 1046个
    trans_name(0)
    # trans_xml()
    # print(fileList[0])
    # tree= parse(os.path.join(path, fileList[0]))
    # print(tree)
    # root = tree.getroot()
    # sub1 = root.find("filename")
    # print(sub1.text)
    # sub1.text = "test"
    # print(sub1.text)
    # tree.write(os.path.join(path, fileList[0]))
