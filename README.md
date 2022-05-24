# Chinese-Text-Classification-Pytorch
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

中文文本分类，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer, 基于pytorch，开箱即用。

## 介绍
模型介绍、数据流动过程：[我的博客](https://zhuanlan.zhihu.com/p/73176084)  

数据以字为单位输入模型，预训练词向量使用 [搜狗新闻 Word+Character 300d](https://github.com/Embedding/Chinese-Word-Vectors)，[点这里下载](https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ)  

## 环境
python 3.7  
pytorch 1.1  
tqdm  
sklearn  
tensorboardX

## 中文数据集
我从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，已上传至github，文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

数据集|数据量
--|--
训练集|18万
验证集|1万
测试集|1万


### 更换自己的数据集
 - 如果用字，按照我数据集的格式来格式化你的数据。  
 - 如果用词，提前分好词，词之间用空格隔开，`python run.py --model TextCNN --word True`  
 - 使用预训练词向量：utils.py的main函数可以提取词表对应的预训练词向量。  


## 效果

模型|acc|备注
--|--|--
TextCNN|91.22%|Kim 2014 经典的CNN文本分类
TextRNN|91.12%|BiLSTM 
TextRNN_Att|90.90%|BiLSTM+Attention
TextRCNN|91.54%|BiLSTM+池化
FastText|92.23%|bow+bigram+trigram， 效果出奇的好
DPCNN|91.25%|深层金字塔CNN
Transformer|89.91%|效果较差
bert|94.83%|bert + fc  
ERNIE|94.61%|比bert略差(说好的中文碾压bert呢)  

bert和ERNIE模型代码我放到另外一个仓库了，传送门：[Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)，后续还会搞一些bert之后的东西，欢迎star。  

## 使用说明
```
# 训练并测试：
# TextCNN
python run.py --model TextCNN

# TextRNN
python run.py --model TextRNN

# TextRNN_Att
python run.py --model TextRNN_Att

# TextRCNN
python run.py --model TextRCNN

# FastText, embedding层是随机初始化的
python run.py --model FastText --embedding random 

# DPCNN
python run.py --model DPCNN

# Transformer
python run.py --model Transformer
```

## 对抗训练使用说明
```
# 训练并测试：
# TextCNN baseline
python run.py --model TextCNN

# TextCNN PGD
python run.py --model TextCNN --adv_module pgd

# TextCNN "Free"
python run.py --model TextCNN --adv_module free

# TextCNN FGSM
python run.py --model TextCNN --adv_module fgsm

```

### 参数
模型都在models目录下，超参定义和模型定义在同一文件中。  

## 对抗实现说明
模块都参考原文伪码实现，对抗模块全部存放与adv_modules,py中，模块记录恢复embedding以及投影的方式参考知乎，差别不会很大，下面详细说明。
### PGD
因为前面只是更新扰动，不想影响模型，所以在pgd中在最后一步之前都会有model.zero_grad()的操作，直到最后一步影响模型更新，同时最后pgd.restore_grad()操作使得原始梯度也能够参与更新。
```
elif adv[0] == "PGD":
    pgd = adv[1]
    K = 3
    pgd.backup_grad()
    # 对抗训练
    for t in range(K):
        pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
        if t != K - 1:
            model.zero_grad()
        else:
            pgd.restore_grad()
        outputs_adv = model(trains)
        loss_adv = F.cross_entropy(outputs_adv, labels)
        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    pgd.restore()  # 恢复embedding参数
```
### FGSM
实现与PGD相似，不同的是这里2步完成，第1步主要是用uniform分布生成初始扰动，加到embedding上。这样得到扰动梯度，更新扰动，第2步真正更新模型参数。
```
def attack(self, epsilon=1., alpha=0.3, emb_name='embedding', is_first_attack=False):
    # emb_name这个参数要换成你模型中embedding的参数名
    for name, param in self.model.named_parameters():
        if param.requires_grad and emb_name in name:
            if is_first_attack:
                self.emb_backup[name] = param.data.clone()
                r_at = param.data.clone()  # uniform初始化
                r_at.uniform_(-epsilon, epsilon)
                param.data.add_(r_at)
                param.data = self.project(name, param.data, epsilon)
            if not is_first_attack:
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)
```
```
elif adv[0] == "FGSM":
    fgsm = adv[1]
    K = 2
    fgsm.backup_grad()
    # 对抗训练
    for t in range(K):
        fgsm.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
        if t != K - 1:
            model.zero_grad()
        else:
            fgsm.restore_grad()
        outputs_adv = model(trains)
        loss_adv = F.cross_entropy(outputs_adv, labels)
        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    fgsm.restore()  # 恢复embedding参数
```
### "Free"
Free最大的不同是模型和扰动一起更新，所以有多次optimizer.step()操作。
```
elif adv[0] == "FREE":
    free = adv[1]
    K = free.free_N
    # 对抗训练
    for t in range(K):
        optimizer.step()
        free.attack(is_first_attack=(t==0))  # 在embedding上添加对抗扰动, first attack时备份param.data
        outputs_adv = model(trains)
        model.zero_grad()
        loss_adv = F.cross_entropy(outputs_adv, labels)
        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    free.restore()  # 恢复embedding参数
```
### 其它
关于embedding是否要恢复是可以讨论的，因为方向传播时计算的实际是扰动后embedding的梯度；这里暂时把embedding当作模型的输入数据，所以都用restore把它进行恢复了。


## 对应论文
[1] Convolutional Neural Networks for Sentence Classification  
[2] Recurrent Neural Network for Text Classification with Multi-Task Learning  
[3] Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification  
[4] Recurrent Convolutional Neural Networks for Text Classification  
[5] Bag of Tricks for Efficient Text Classification  
[6] Deep Pyramid Convolutional Neural Networks for Text Categorization  
[7] Attention Is All You Need  
