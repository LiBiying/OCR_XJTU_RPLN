说明文档（By XJTU_RPLN）
=========
## 数据处理
### 数据集合并
手动把初赛和复赛的数据集合并，增大数据集的规模，目的是为了增加网络泛化能力。
### 数据划分
按照0.15的比值划分出验证集。数据集划分代码见./code/data_split.py
### 数据处理
**无**数据扩增（旋转、噪声等）操作，直接将RGB图片转为灰度图像之后按照网络输入要求进行大小变换，主要操作是图片补零。<br>
考虑到数据集中长256宽48的图片占多数，为了节省计算资源，直接以此长宽为标准。即如果图片长宽是（256,48）那么不对该图片大小做变换。否则根据长宽比，过长的直接resize到256长度，宽补零；过宽的直接resize到48宽度，长补零。<br>
数据处理代码见./code/ds.py
## 模型训练
### 基础模型
基础模型从论文《An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition》复现。论文链接：https://arxiv.org/abs/1507.05717<br>
该模型是一个端到端的文字识别模型，由三部分构成：首先将处理之后的图片送入CNN，经过7层卷积提取特征，然后将特征图按照宽度不变，长度为1的形式作为序列输入双向2层LSTM进行学习，学习到的结果使用CTC结构化损失，反向传递调整网络参数。
基础模型示意图如下：<br>
<div align=center><img width="300" height="400" src="https://github.com/LiBiying/OCR_XJTU_RPLN/raw/master/网络示意图.JPG"/></div><br>
基础模型结构为：<br>
<div align=center><img width="300" height="300" src="https://github.com/LiBiying/OCR_XJTU_RPLN/raw/master/网络结构.JPG"/></div><br>

### 模型改进
在调参过程中，主要对基础模型进行了四方面调整。由于训练准确度始终比验证准确度高不少，判断出现了过拟合，因此改动主要围绕减小过拟合进行。<br>
第一处是CRNN中CNN部分的通道数，原网络每层通道数[64,128,256,256,512,512,512]，改为[64,128,128,256,256,512,512]<br>
第二处是在每层卷积层后都加入BN<br>
第三处是尝试了不同激活函数<br>
第四处是对LSTM采用dropout<br>
最终按照ensemble方法，将几个效果较好的模型进行投票得出预测结果。<br>
模型代码见./code/crnn.py<br>
模型训练代码见./code/train.py<br>
模型及参数见./models，其中crnn.pkl是网络结构，其余文件都是较好模型结果，都是在修改CNN通道数并加上BN之后调整激活函数和dropout比例得到的。

