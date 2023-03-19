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