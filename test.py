import numpy as np
import torch
import torch.nn as nn
from img_dataset import return_dataloader
from model import SRNet
import os
from train import val
from cfg import cfg
from sklearn import metrics

log_dir = cfg['log_dir']

def test(mdoel,device, criterion, test_set, test_set_len, batch_size):
    test_acc_num = 0
    test_loss = 0.0
    test_num =  0
    y_true = []
    y_pre = []
    for index, data in enumerate(test_set):
        test_num += 1
        model.eval()
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            pre = model(inputs)

            _, pre_index = torch.max(pre, 1)
            pre_index = pre_index.reshape(-1, 1)
            test_acc_num += (pre_index.cpu() == labels.cpu()).sum().item()

            labels = torch.squeeze(labels)
            loss = criterion(pre, labels)
            test_loss += loss

            for i in range(0,batch_size):
                y_true.append(labels[i].cpu().item())
                y_pre.append(pre_index[i].cpu().item())



    return test_acc_num/test_set_len, test_loss/test_num, y_true, y_pre

if __name__ == '__main__':
    device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
    model = SRNet(1)
    state_dict = torch.load(log_dir+'/model_75.ckpt')
    model.load_state_dict(state_dict)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_set, test_len = return_dataloader(cfg['cover_dir'], cfg['stego_dir'],
                                           cfg['test_file'], 'test',
                                           cfg['test_batch_size'])
    print("Test begin......")
    test_acc, test_loss, y_true, y_pre = test(model,device,criterion,test_set,test_len,cfg['test_batch_size'])

    confusion_matrix = metrics.confusion_matrix(y_true, y_pre)
    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TP = confusion_matrix[1][1]
    train_acc2 = (TN+TP)/(TN+TP+FN+FP)

    print("test acc is {:.3f}, loss is {:.3f}".format(test_acc, test_loss))
