"# ClickAI_scratch3" 
"# ClickAI_scratch3" 
# Codelab_clickAI🔥



# 👉“Click Here，Create Here”
为了能够让scratch与AI的相关功能结合起来，使得通过图形化编程的方式能够调用更多有趣的AI功能（机器视觉、机器语音、自然语言理解以及机器人和AR的相关功能），而不需要关注其复杂的内部实现，激发孩子们的创造力，我们基于codelab开发了这款插件。同时，这也有利于将scratch作为桥梁把AI和智能硬件（如树莓派、arduino、jeton nano）连接起来，让孩子们能够轻松开发各类炫酷的AI硬件（后续我们会陆续上传demo并开放源代码），不需要复杂的算法和代码知识，在这里，点击即是创造。




# 📰使用方法
codelab_ClickAI的插件包含以下几个部分的功能，后续也会陆续的完善和增加：
## 机器视觉👀
### 1.图像分类(codelab_classifier.py)
```bash
pip install baidu-aip
pip install numpy
pip install opencv-python
pip install codelab_adapter_client --upgrade
```
🕐安装完上述依赖后运行codelab_adapter3.x
（暂时我们的extension没有UI，所以需要手动运行我们的python文件）

🕑进入到codelab_classifier.py所在的文件夹
```bash
python3 codelab_classifier.py
```
🕒打开adapter UI中的scratch，加载EIM插件（暂时我们的extension没有UI，所以先借用EIM），发送话题为eim/classifier的消息，内容为需要进行图像分类的本地图片路径（如D:\image.jpg） 




### 2.目标检测
```bash
pip install pycocotools numpy opencv-python tqdm tensorboard tensorboardX pyyaml webcolors
pip install torch==1.4.0
pip install torchvision==0.5.0
pip install codelab_adapter_client --upgrade
```
🕐安装完上述依赖后在[https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git)下载efficientdet源码
进入文件夹后将[codelab_obj_detection_effdet.py](https://github.com/pigtigger/ClickAI_scratch3/blob/master/codelab_obj_detection_effdet.py)复制过来
```bash
python3 codelab_obj_detection_client.py
```
codelab_adapter3.x
（暂时我们的extension没有UI，所以需要手动运行我们的python文件）


🕑运行codelab_adapter3.x


🕒打开adapter UI中的scratch，加载EIM插件（暂时我们的extension没有UI，所以先借用EIM），发送话题为eim/obj_detection的消息，内容为需要进行目标检测的物体种类，（如person、bottle），目前仅提供以下类别
```python
#所有支持识别的种类
['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']
```
 🕓可以从收到的消息积木块中获取所识别到物体的中心位置坐标，进而与其他功能进行结合
比如控制舞台小猫跟随目标物体移动
### 3.图片文字识别
```bash
pip install baidu-aip
pip install numpy
pip install opencv-python
pip install codelab_adapter_client --upgrade
```
🕐安装完上述依赖后运行codelab_adapter3.x
（暂时我们的extension没有UI，所以需要手动运行我们的python文件）


🕑进入到codelab_classifier.py所在的文件夹
```bash
python3 codelab_imag2txt.py
```
🕒打开adapter UI中的scratch，加载EIM插件（暂时我们的extension没有UI，所以先借用EIM），发送话题为eim/img2txt的消息，内容为需要进行文本提取的本地图片路径（如D:\image.jpg） 

🕓可以从收到的消息积木块中获取所识别到的文本，进而与其他功能进行结合
### 4.人脸/手势识别
```bash
pip install baidu-aip
pip install numpy
pip install opencv-python
pip install codelab_adapter_client --upgrade
```
🕐安装完上述依赖后运行codelab_adapter3.x
（暂时我们的extension没有UI，所以需要手动运行我们的python文件）


🕑进入到codelab_classifier.py所在的文件夹
```bash
python3 codelab_pose.py
```
🕒打开adapter UI中的scratch，加载EIM插件（暂时我们的extension没有UI，所以先借用EIM），发送话题为eim/pose的消息，内容为pose或face（pose为进行手势识别，face为进行人脸识别） 


🕓可以从收到的消息积木块中获取所识别到的手势名称或手势数量，进而与其他功能进行结合
### 5.图像分割
（还在完善）
### ........
## 机器语音🎤
（还在完善）
### 1.语音识别
### 2.文本转语音
### 3.图灵机器人语义理解
### .........
## 机器人👾
（还在完善）
### 1.模拟扫地机器人建图
### 2.模拟扫地机器人自主导航
### 3.语音交互机器人
## 增强现实(AR)&虚拟现实(VR)🎮
（还在完善）
### 1.在现实环境中放置物体
### ..........
## 
