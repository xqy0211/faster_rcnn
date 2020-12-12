import os
import random
"""
根据标注文件annotation 生成train.txt和val.txt文件
"""

files_path = r"G:\xqy\faster_rcnn\TGK_DATASET\Annotations"      # 保存所有xml标注文件的根目录
if not os.path.exists(files_path):
    print("文件夹不存在")
    exit(1)
val_rate = 0.2      # 验证集的比例

files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])    # 遍历分割并排序
files_num = len(files_name)
val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
train_files = []
val_files = []
for index, file_name in enumerate(files_name):
    if index in val_index:
        val_files.append(file_name)
    else:
        train_files.append(file_name)

try:
    train_f = open("train.txt", "x")
    eval_f = open("val.txt", "x")
    train_f.write("\n".join(train_files))
    eval_f.write("\n".join(val_files))
    print("根目录下生成文件：train.txt和val.txt，划分比例为[{:.0f}:{:.0f}]请放到对应Main文件夹中！".format(10-val_rate*10, val_rate*10))
except FileExistsError as e:
    print(e)
    exit(1)



