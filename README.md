说明文档（By XJTU_RPLN）
=========
## 数据处理
### 数据集合并
手动把初赛和复赛的数据集合并，增大数据集的规模，目的是为了增加网络泛化能力。
### 数据划分
按照0.15的比值划分训练集和验证集。
### 颜色处理
直接RGB转灰度图像。
### 图片补零
考虑到数据集中长256宽48的图片最多，所以以此为标准，如果长宽是（256,48）那么大小不变。否则根据长宽比，过长的直接resize到256长度，宽补零；过宽的直接resize到48宽度，长补零。
## 模型训练
### 基础模型
基础模型从论文《An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition》复现。论文链接：https://arxiv.org/abs/1507.05717<br>
该模型是一个端到端的文字识别模型，由三部分构成：首先将处理之后的图片送入CNN，经过6层卷积提取特征，然后将特征图按照宽度不变，长度为1的形式作为序列输入双向LSTM进行学习，学习到的结果使用CTC结构化损失，反向传递调整网络参数。基础模型示意图见图1，基础模型结构见表1。
![](https://github.com/LiBiying/OCR_XJTU_RPLN/raw/master网络结构.jpg)
### 模型改进
改：CRNN中CNN部分的通道数<br>
改：激活函数<br>
加：BN层<br>
模型代码见./code/crnn.py
