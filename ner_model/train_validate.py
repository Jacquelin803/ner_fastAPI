from utils import *
from BiLSTM_CRF import *
from config import *
import torch

if __name__ == '__main__':

    dataset_train = Dataset()
    loader_train = data.DataLoader(
        dataset_train,
        batch_size=100,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dataset_test = Dataset('test')
    loader_test = data.DataLoader(dataset_test, batch_size=100, collate_fn=collate_fn)

    model = BILSTM_CRF().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total = len(dataset_train)

    for e in range(EPOCH):
        for b, (input, target, mask) in enumerate(loader_train):
            y_pred = model(input, mask)

            loss = model.loss_fn(input, target, mask).to(DEVICE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 100 == 0:

                # print('>> epoch:', e, 'train-loss:', loss.item())
                print('>> epoch:', e, ' b', b, 'total', total, 'loss:', loss.item())

        with torch.no_grad():
            y_true_list = []
            y_pred_list = []
            total_test=len(loader_test)
            for b, (input, target, mask) in enumerate(loader_test):
                y_pred = model(input, mask)
                loss = model.loss_fn(input, target, mask)

                # 拼接返回值
                for lst in y_pred:
                    y_pred_list += lst
                for y, m in zip(target, mask):
                    y_true_list += y[m == True].tolist()

            # 整体准确率
            y_true_tensor = torch.tensor(y_true_list)
            y_pred_tensor = torch.tensor(y_pred_list)
            accuracy = (y_true_tensor == y_pred_tensor).sum() / len(y_true_tensor)
            print('>> total:', len(y_true_tensor), 'accuracy:', accuracy.item())

        # torch.save(model, MODEL_DIR + f'model_{e}.pth')
    # torch.save(model, MODEL_DIR + 'model.pth')
    torch.save(model.state_dict(), MODEL_DIR + 'model1.pth')