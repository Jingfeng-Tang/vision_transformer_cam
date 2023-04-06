# vision_transformer_cam
use vit to generate class activate map
## 2023年3月17日20:16:09 
add-----使用pascal voc2012数据集训练\
add-----训练时生成cam图
## 2023年3月18日15:29:05
add-----利用cam图生成pseudo seg result
## 2023年3月18日19:59:43
add-----train and validate 完成，其中validate时计算mAP\
mod-----train.py为train_and_validate.py
## 2023年3月18日21:32:16
add-----命名为training_log_datatime.txt的文本日志\
mod-----seed函数，使用更加全面的随机种子函数\
issue-----训练时验证mAP达到1，但是使用验证集中数据进行验证发现错很多，说明average precision函数用的不对，而且模型还存在问题
## 2023年3月18日21:59:43
add-----train时，模型输出的pred 经过softmax再进行损失计算，看看效果会不会好
## 2023年3月19日09:25:29
softmax应该是对的\
issue-----上版本训练，loss只到0.697，mAP 0.65 左右，是否需要更多epoch（当前500）或者调整训练参数？\
add-----shell脚本，需输入参数\
add-----update_log.md， 更新日志从readme中分离\
mod-----完善参数部分代码，大部分参数required=True
mod-----模型权重加载时存在bug
## 2023年3月19日20:47:32
issue-----设置500ep,batchsize32|700ep,bs32|1000ep,8bs 效果都不好mAP在0.65左右，loss在0.7左右降不下去，搜了一下感觉应该用sigmoid做激活函数，试一下吧。
## 2023年3月20日14:51:24
add-----遵从ToCo与MCTformer的实验设置，使用ImageNet预训练权重训练模型\
add-----确定两个评估指标\
1、mAP，用于多标签分类，在验证阶段和测试阶段\
2、mIOU，用于语义分割，在验证阶段和测试阶段\
目前mAP在验证阶段已经写完测试完，测试了sklearn的average_precision_score，该函数有两个输入outputs、labels\
模型输出outputs后，加上sigmoid，将特征映射到（0,1）后为函数输入的outputs\
labels为真实标签，例子(1, 0, 1, 0, 0, 0, 1)
## 2023年3月20日20:38:26
mod-----找到了validate时mAP过低的原因，在预加载权重时删除了分类头权重，导致多标签分类不准，修改后mAP达到0.876\
mod-----utils中validate中mAP的单步和总体均值显示\
add-----validate.py mAP计算\
add-----data.py 分割标签的导入
## 2023年3月21日15:20:25
add-----validate.py seg_res生成分割图，类别颜色标识都是对的，但效果很差\
## 2023年3月22日09:36:23
add-----validate.py high-threshold最佳0.916 pretrained -ep1000
## 2023年3月23日16:57:22
mod-----validate.py predict.py添加注意力map，效果较好，但是缺乏类别信息
## 2023年3月24日17:08:28
add-----分布式训练
## 2023年3月28日08:14:46
add-----Vit的每个block的attn_weights, 原图, syn_weights可视化
## 2023年3月30日16:23:14
add-----predict.py 添加same_seed，保持种子一致
## 2023年4月2日20:56:54
add-----昨天已解决注意力再汇聚的问题，在ViT中添加了找类别的思路，先存一版。
## 2023年4月5日11:41:26
add-----vit test
## 2023年4月5日20:55:04
add-----能够给特征图赋予类别了
## 2023年4月6日15:25:59
add-----使用16hwp与所有patch进行相似度计算，生成相似度矩阵，再插值回原图\
add-----使用block5的注意力权重，插值，进行前景与背景的分割