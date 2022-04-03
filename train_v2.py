import numpy as np
import torch
import torch.nn as nn
from img_dataset3 import return_dataloader
from model import SRNet
import os
from cfg import cfg
#--------------------------------tensorboard--------------------------------------
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(cfg['tensorboard_log_dir'])
#---------------------------------------------------------------------------------

def train(model,optimizer,device,criterion,tr_set, tr_set_len):
    train_acc_num = 0
    train_loss = 0.0
    train_num = 0
    for index,data in enumerate(tr_set):
        train_num += 1
        model.train()
        optimizer.zero_grad()
        # inputs, labels = data

        inputs_pairs, labels_pairs = data
        inputs_cover, inputs_stego = inputs_pairs.split(1, 1)
        labels_cover, labels_stego = labels_pairs.split(1, 1)
        inputs = torch.cat((inputs_cover,inputs_stego), 0)
        labels = torch.cat((labels_cover, labels_stego), 0)

        inputs, labels = inputs.to(device), labels.to(device)
        pre = model(inputs)

        _, pre_index = torch.max(pre, 1)
        pre_index = pre_index.reshape(-1,1)
        train_acc_num += (pre_index.cpu() == labels.cpu()).sum().item()

        labels = torch.squeeze(labels)
        loss = criterion(pre, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().item()

    return train_acc_num/tr_set_len, train_loss/train_num


def val(model,device,criterion,val_set, val_set_len):
    val_acc_num = 0
    val_loss = 0.0
    val_num =  0
    for index, data in enumerate(val_set):
        val_num += 1
        model.eval()
        with torch.no_grad():
            # inputs, labels = data

            inputs_pairs, labels_pairs = data
            inputs_cover, inputs_stego = inputs_pairs.split(1, 1)
            labels_cover, labels_stego = labels_pairs.split(1, 1)
            inputs = torch.cat((inputs_cover, inputs_stego), 0)
            labels = torch.cat((labels_cover, labels_stego), 0)

            inputs, labels = inputs.to(device), labels.to(device)
            pre = model(inputs)

            _, pre_index = torch.max(pre, 1)
            pre_index = pre_index.reshape(-1, 1)
            val_acc_num += (pre_index.cpu() == labels.cpu()).sum().item()

            labels = torch.squeeze(labels)
            loss = criterion(pre, labels)
            val_loss += loss

    return val_acc_num/val_set_len, val_loss/val_num

log_dir = cfg['log_dir']
model_path = log_dir+'/model_'+str(cfg['img_qc'])+'.ckpt'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)




if __name__ == '__main__':
    device = 'cuda:5'
    model = SRNet(1)
    model.to(device)

#------------------------------init----------------------------------------
    for m in model.modules():
          if isinstance(m, (nn.Conv2d)):
                 nn.init.kaiming_normal_(m.weight, mode='fan_in')
                 nn.init.constant_(m.bias, val=0.2)
          elif isinstance(m, (nn.Linear)):
                 nn.init.normal_(m.weight, mean=0, std=0.01)

#--------------------------------------------------------------------------

    params = [{'params': (p for name,p in model.named_parameters() if ('conv' in name) and ('weight' in name)),
               'weight_decay': 2e-4},
              {'params': (p for name,p in model.named_parameters() if ('conv' not in name) or ('weight' not in name)),
               }]

    optimizer = torch.optim.Adamax(params, lr=cfg['learning_rate'])

    criterion = nn.CrossEntropyLoss()
    print("Using imgdataset3")
    print("Loading Data")

    train_set, train_len = return_dataloader(cfg['cover_dir'],
                                 cfg['stego_dir'],
                                 cfg['train_file'],
                                 type='train',
                                 batch_size=cfg['train_batch_size']      #train dataloader

    valid_set, valid_len = return_dataloader(cfg['cover_dir'],
                                  cfg['stego_dir'],
                                  cfg['valid_file'],
                                  type='valid',
                                  batch_size=cfg['valid_batch_size'])    #valid_dataloader

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['epochs_boundary'],
                                                gamma=0.1)

    max_acc = 0                               #model's best accuracy

# -----------------------------tensorboard-----------------------------------------
    input_sample = torch.zeros((1, 1, 256, 256), dtype=torch.float)
    input_sample = input_sample.to(device)
    writer.add_graph(model, input_sample)
# ---------------------------------------------------------------------------------


    for epoch in range(0,cfg['epochs']):
        train_acc, train_loss = train(model,optimizer,device,criterion,
                                      train_set,train_len)     #train

        scheduler.step()

        val_acc, val_loss = val(model,device,criterion,
                                valid_set, valid_len)          #validation

#----------------------------print-------------------------------------------------
        if (epoch+1) <= 10:
            print("Epoch:{}, train acc is {:.3f}, loss is {:.3f}".format(epoch+1, train_acc, train_loss))
            print("val acc is {:.3f}, loss is {:.3f}".format(val_acc, val_loss))
            print("")
        else:
            if (epoch+1)%20 == 0:
                print("Epoch:{}, train acc is {:.3f}, loss is {:.3f}".format(epoch+1, train_acc, train_loss))
                print("val acc is {:.3f}, loss is {:.3f}".format(val_acc, val_loss))
                print("")
            elif epoch+1 == cfg['epochs']:
                print("Training End!!! ")
                print("train acc is {:.3f}, loss is {:.3f}".format(epoch + 1, train_acc, train_loss))
                print("val acc is {:.3f}, loss is {:.3f}".format(val_acc, val_loss))
                print("")
#-----------------------------------------------------------------------------------
        if epoch>457:
            if max_acc<val_acc:                #save best model
                max_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('Save model_qc{} {}'.format(cfg['img_qc'],epoch+1))

        # with open(log_dir + '/LogFile/valid_acc.csv', 'a+') as f_val:
        #     f_val.write('%d,%.4f,%.4f\n' % (epoch+1, val_loss, val_acc))

#-----------------------------tensorboard-----------------------------------------
        writer.add_scalar('train_acc', train_acc, global_step=epoch)
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
#---------------------------------------------------------------------------------



