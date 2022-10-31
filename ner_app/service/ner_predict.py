from ner_app.service.BiLSTM_CRF import BILSTM_CRF
from ner_app.service.utils import *


# GPU训练的模型，布在GPU服务器上


class NER_Predict:

    def __init__(self):

        self.info=[]


    def predict(self,text):
        _, word2id = get_vocab()
        input = torch.tensor([[word2id.get(w, WORD_UNK_ID) for w in text]])
        mask = torch.tensor([[1] * len(text)]).bool()

        model = BILSTM_CRF()
        model.load_state_dict(torch.load(MODEL_DIR + 'model1.pth'))
        # model = torch.load(MODEL_DIR + 'model1.pth')
        y_pred = model(input, mask)
        id2label, _ = get_label()

        label = [id2label[l] for l in y_pred[0]]
        print(text)
        print(label)

        info = extract(label, text)
        print(info)
        print(text)
        # info="red,df"
        print("=======================",info)
        return info


if __name__ == '__main__':
    text = '每个糖尿病患者,无论是病情轻重,不论是注射胰岛素,还是口服降糖药,都必须合理地控制饮食。'
    NER_Predict().predict(text)