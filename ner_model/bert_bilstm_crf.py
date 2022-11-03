import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BertModel
from torch.optim import AdamW
import torch.nn as nn

from config import *
from torchcrf import CRF

from seqeval.metrics import accuracy_score as seq_accuracy_score
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import precision_score as seq_precision_score
from seqeval.metrics import recall_score as seq_recall_score



def read_data(file):
    with open(file,"r",encoding="utf-8") as f:
        all_data = f.read().split("\n")

    all_text = []
    all_label = []

    text = []
    label = []
    for data in all_data:
        if data == "":
            all_text.append(text)
            all_label.append(label)
            text = []
            label = []
        else:
            t,l = data.split(" ")
            text.append(t)
            label.append(l)

    return all_text,all_label

# 将label转化为index标签，因为预测label全是按照index,最后label_2_index会包含所有的预测标签
def build_label(train_label):
    label_2_index = {"PAD":0,"UNK":1}
    for label in train_label:
        for l in label:
            if l not in label_2_index:
                label_2_index[l] = len(label_2_index)
    return label_2_index,list(label_2_index)


class BertDataset(Dataset):
    def __init__(self,all_text,all_label,label_2_index,max_len,tokenizer):
        self.all_text = all_text
        self.all_label = all_label
        self.label_2_index = label_2_index
        self.tokenizer = tokenizer
        self.max_len = max_len
    # 在走到dataloader调用for机构时就会运行
    def __getitem__(self,index):
        text = self.all_text[index]
        label = self.all_label[index][:self.max_len]
        # 句子中的每个字在vocab.txt里的映射成的index,句子里字数大于30的部分截掉，小于30的用0补齐
        # 这种会造成一定数据浪费，还有一种写法是每过max_len就形成一个句子，但损失了原本句子完整性
        # 会将这两一头一尾加上去: max_length=self.max_len + 2
        # 101 [UNK]
        # 102 [CLS]
        text_index = self.tokenizer.encode(text,add_special_tokens=True,max_length=self.max_len+2,padding="max_length",truncation=True,return_tensors="pt")
        label_index = [0] +  [self.label_2_index.get(l,1) for l in label] + [0] + [0] * (self.max_len - len(text))

        label_index = torch.tensor(label_index)
        return  text_index.reshape(-1),label_index,len(label)

    def __len__(self):
        return self.all_text.__len__()



class BertBilstmNerModel(nn.Module):
    def __init__(self,lstm_hidden,class_num):
        super().__init__()

        # 这个就是将bert输出的embedding嫁接进来
        self.bert = BertModel.from_pretrained(os.path.join("./output/","bert_base_chinese"))
        for name,param in self.bert.named_parameters():
            param.requires_grad = False

        self.lstm=nn.LSTM(768,lstm_hidden,batch_first=True,num_layers=2,bidirectional=False)

        self.classifier = nn.Linear(lstm_hidden,class_num)

        self.crf=CRF(class_num,batch_first=True)  # loss_fun in CRF

        # self.loss_fun = nn.CrossEntropyLoss()

    def forward(self,batch_index,batch_label=None):
        bert_out = self.bert(batch_index)
        bert_out0,bert_out1 = bert_out[0],bert_out[1]
        # bert_out0:字符级别特征, bert_out1:篇章级别
        # 字符级别用于ner这类以文字为最小判别单位的任务，篇章级别针对文本分类这种以一段话为最小判别单位的任务

        lstm_out,_= self.lstm(bert_out0)

        pred = self.classifier(lstm_out)
        # pred=pred.reshape(-1, pred.shape[-1])
        # batch_label=batch_label.reshape(-1)
        if batch_label is not None:
            # loss = self.loss_fun(pred,batch_label)
            loss=-self.crf(pred,batch_label)
            return loss
        else:
            pred=self.crf.decode(pred)
            return pred


if __name__ == '__main__':
    train_text, train_label = read_data(os.path.join("./output/", "train.txt"))
    # train_text, train_label = read_data(os.path.join("./output/", "test_sample2.txt"))
    dev_text, dev_label = read_data(os.path.join("./output/", "dev.txt"))
    # test_text, test_label = read_data(os.path.join("./output/", "test.txt"))

    label_2_index, index_2_label = build_label(train_label)
    # label_2_index:
    # {'PAD': 0, 'UNK': 1, 'B-NAME': 2, 'E-NAME': 3, 'O': 4, 'B-CONT': 5, 'M-CONT': 6, 'E-CONT': 7, 'B-RACE': 8, 'E-RACE': 9, 'B-TITLE': 10, 'M-TITLE': 11, 'E-TITLE': 12, 'B-EDU': 13, 'M-EDU': 14, 'E-EDU': 15, 'B-ORG': 16, 'M-ORG': 17, 'E-ORG': 18, 'M-NAME': 19, 'B-PRO': 20, 'M-PRO': 21, 'E-PRO': 22, 'S-RACE': 23, 'S-NAME': 24, 'B-LOC': 25, 'M-LOC': 26, 'E-LOC': 27, 'M-RACE': 28, 'S-ORG': 29}

    tokenizer = BertTokenizer.from_pretrained(os.path.join("./output/", "bert_base_chinese"))

    batch_size = 100
    epoch = 100
    max_len = 30
    lr = 0.0005
    lstm_hidden=128
    train_dataset = BertDataset(train_text, train_label, label_2_index, max_len, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    dev_dataset = BertDataset(dev_text, dev_label, label_2_index, max_len, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    # train_dataloader

    model = BertBilstmNerModel(lstm_hidden,len(label_2_index)).to(DEVICE)
    opt = AdamW(model.parameters(), lr)

    for e in range(epoch):
        model.train()
        # batch_text_index: [50,32] 50句话，每句话32个字，当然其中有些字是padding
        for batch_text_index, batch_label_index, batch_len in train_dataloader:
            batch_text_index = batch_text_index.to(DEVICE)
            batch_label_index = batch_label_index.to(DEVICE)
            loss=model.forward(batch_text_index,batch_label_index)
            loss.backward()

            opt.step()
            opt.zero_grad()

            print(f'loss: {loss:.2f}')

        model.eval()

        all_pred=[]
        all_targ=[]
        for batch_text_index, batch_label_index, batch_len in dev_dataloader:
            batch_text_index = batch_text_index.to(DEVICE)
            # batch_label_index = batch_label_index.to(DEVICE)
            pred=model.forward(batch_text_index)

            # pred=pred.cpu().numpy().tolist()
            targ=batch_label_index.numpy().tolist()

            for p,t,l in zip(pred,targ,batch_len):
                p=p[1:1+l]
                t=t[1:1+l]

                p=[index_2_label[i] for i in p]
                t=[index_2_label[i] for i in t]

                all_pred.append(p)
                all_targ.append(t)

        f1_score = seq_f1_score(all_targ, all_pred)
        acc=seq_accuracy_score(all_targ,all_pred)
        print(f"f1:{f1_score}, acc:{acc}")

