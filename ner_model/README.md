### Pytorch BiLSTM_CRF医疗命名实体识别

本项目是阿里天池大赛的一个经典赛题，《瑞金医院MMC人工智能辅助构建知识图谱大赛》，赛题要求选手在糖尿病相关的学术论文和临床指南的基础上，做实体的标注，也就是NLP领域常说的，命名实体识别（Named Entity Recognition, NER）任务。

天池赛题地址：https://tianchi.aliyun.com/competition/entrance/231687/information

crf部分直接引源码包，不用自己手写loss

只训练了3个epoch，大概30分钟左右
>> epoch: 0  b 0 total 103441 loss: 172.92007446289062
>> epoch: 0  b 100 total 103441 loss: 49.18278121948242
>> epoch: 0  b 200 total 103441 loss: 46.93279266357422
>> epoch: 0  b 300 total 103441 loss: 46.27277755737305
>> epoch: 0  b 400 total 103441 loss: 43.51652145385742
>> epoch: 0  b 500 total 103441 loss: 34.710411071777344
>> epoch: 0  b 600 total 103441 loss: 37.08740234375
>> epoch: 0  b 700 total 103441 loss: 35.45085525512695
>> epoch: 0  b 800 total 103441 loss: 34.5488166809082
>> epoch: 0  b 900 total 103441 loss: 30.40422821044922
>> epoch: 0  b 1000 total 103441 loss: 28.245849609375
>> total: 1152462 accuracy: 0.8365933299064636
>> epoch: 1  b 0 total 103441 loss: 28.626258850097656
>> epoch: 1  b 100 total 103441 loss: 27.023632049560547
>> epoch: 1  b 200 total 103441 loss: 23.992084503173828
>> epoch: 1  b 300 total 103441 loss: 23.852821350097656
>> epoch: 1  b 400 total 103441 loss: 19.729625701904297
>> epoch: 1  b 500 total 103441 loss: 21.856426239013672
>> epoch: 1  b 600 total 103441 loss: 18.52393913269043
>> epoch: 1  b 700 total 103441 loss: 18.97027015686035
>> epoch: 1  b 800 total 103441 loss: 22.58467674255371
>> epoch: 1  b 900 total 103441 loss: 17.874914169311523
>> epoch: 1  b 1000 total 103441 loss: 21.172943115234375
>> total: 1152462 accuracy: 0.8692625164985657
>> epoch: 2  b 0 total 103441 loss: 19.239727020263672
>> epoch: 2  b 100 total 103441 loss: 17.90894889831543
>> epoch: 2  b 200 total 103441 loss: 18.60819435119629
>> epoch: 2  b 300 total 103441 loss: 19.005260467529297
>> epoch: 2  b 400 total 103441 loss: 18.889728546142578
>> epoch: 2  b 500 total 103441 loss: 17.370134353637695
>> epoch: 2  b 600 total 103441 loss: 15.502437591552734
>> epoch: 2  b 700 total 103441 loss: 19.936967849731445
>> epoch: 2  b 800 total 103441 loss: 15.709561347961426
>> epoch: 2  b 900 total 103441 loss: 15.289838790893555
>> epoch: 2  b 1000 total 103441 loss: 18.290386199951172
>> total: 1152462 accuracy: 0.8816403746604919

建议用GPU训练，本地训练该项目可能需要几个小时。本代码已改好，既可以本地也可以GPU

pytorch的代码 CPU改为GPU找一下几个地方改动

一、网络模型

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mymodule = MyModule()
# mymodule = mymodule.cuda() # 不建议用这种
mymodule = mymodule.to(DEVICE)

二、数据（输入、标记）

        images,targets = data
        images = images.to(DEVICE)
        targets = targets.to(DEVICE) #训练数据和测试数据集都要进行此操作

三、损失函数

# 损失函数
loss = nn.CrossEntropyLoss()
loss = loss.to(DEVICE)
