# Faster R-CNN
## 环境配置：
* Python3.6
* Pytorch1.6
* pycocotools(Linux: pip install pycocotools;   
  Windows:pip install pycocotools-windows(不需要额外安装vs))

## 文件结构：
```
* ├── backbone: 特征提取网络，可以根据自己的要求选择
* ├── network_files: Faster R-CNN网络（包括Fast R-CNN以及RPN等模块）
* ├── train_utils: 训练验证相关模块（包括cocotools）
* ├── my_dataset.py: 自定义dataset用于读取VOC数据集
* ├── train_mobilenet.py: 以MobileNetV2做为backbone进行训练
* ├── train_resnet50_fpn.py: 以resnet50+FPN做为backbone进行训练
* ├── train_multi_GPU.py: 针对使用多GPU的用户使用
* ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
* ├── pascal_voc_classes.json: pascal_voc标签文件
```

## 数据集
* Pascal VOC2012 train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
* PCB_DATASET：参考https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB
* DAGM_DATASET：参考https://www.pythonf.cn/read/127802
* 数据标注：


## 项目流程

### Step1 确保提前准备好数据集
* 首先的数据的采集（网上找或者拍照）与增强（用CV方法进行P图或者用SinGAN生成训练样本），之后进行标注
* 标注选用常见的VOC2012格式，文件结构如下，不用管为啥这么排：  
```
数据集文件夹名称(如FPCcrease_DATASET)
* ├── Annotations: 存放样本图像的标注文件信息
* ├── ImageSets
    ├──Main 
        ├──train.txt：内部每一行都是一个训练样本的名称
        ├──test.txt：内部每一行都是一个验证样本的名称
* ├── JPEGImages: 存放样本图像
```  
* 标注工具在Anaconda环境的pytorch16中，使用`conda activate pytorch16`激活环境后，键入`labelimg`启动标注工具

![](image/labelimg初始化界面.png)

1、先点击左侧Change Save Dir 选择要保存标注文件的路径，如(FPCcrease_DATASET/Annotations)，  
2、之后点击Open Dir选择要标注的样本图片，如(FPCcrease_DATASET/JPEGImages)  
3、之后进行标注，标注的时候应该尽可能细，因为这是监督信息，根本上决定了训练的模型好坏，就按照我们希望它识别出的结果来标，有一定标准性  

* 标注完成后，还缺少Main文件下的两个txt文件，这里需要用本项目下的`spilt_data.py`脚本生成，运行脚本前，请修改该脚本的标注文件根目录，并按我们的期望划分训练集和验证集（小样本2：8就行，测试集需要在划分前挑选出少量放到一边）
* 新建存放具体类别的**字典**（取决于标注时的输入），如`fpc_dataset.json`
* 生成txt完成后，需要新建继承了Pytorch的DataSet的数据集文件，如`fpc_dataset.py`

## Step2 确保提前下载好对应预训练模型权重（下载后放入backbone文件夹中）
* MobileNetV2 backbone: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

## Step3 训练前期的路径及参数调整
* `train_res50_fpn.py`中需要导入我们自己定义的数据集，进行部分路径的修改，主要参数修改：num_classes,lr,lr调整策略,总epoch数和batch_size  
* 另一个要修改的参数主要在`network_files/faster_rcnn_framework.py`中，FasterRCNN类下的proposal数,NMS阈值

## Step4 训练模型
* `conda activate pytorch16`
* 若要训练mobilenetv2+fasterrcnn，直接使用`python train_mobilenet.py`训练脚本
* 若要训练resnet50+fpn+fasterrcnn，直接使用`python train_resnet50_fpn.py`训练脚本
* 若要使用多GPU训练，使用```python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py```指令,```nproc_per_node```参数为使用GPU数量

# 学习自
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing