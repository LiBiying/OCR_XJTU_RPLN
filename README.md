说明文档（By XJTU_RPLN）
=========
## 数据处理
### 数据集合并
我们首先把初赛和复赛的数据集手动合并，增大数据集的规模。
### 数据划分
按照0.15的比值划分训练集和验证集。
### 颜色处理
直接RGB转灰度图像。
### 图片补零
考虑到数据集中长256宽48的图片最多，所以以此为标准，如果长宽是（256,48）那么大小不变。否则根据长宽比，过长的直接resize到256长度，宽补零；过宽的直接resize到48宽度，长补零。
## 参考模型
基础模型从论文arXiv:1507.05717 [cs.CV]复现。
## 模型改进
改：CRNN中CNN部分的通道数<br>
改：激活函数<br>
加：BN层<br>
