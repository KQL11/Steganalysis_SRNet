import scipy.io as sio
import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from random import random as rand
from cfg import cfg


def rot_flip_pair(img):
    rot = random.randint(0,3)
    if rand()<0.5:
        img[0] = np.rot90(img[0], rot, axes=[0,1])
        img[1] = np.rot90(img[1], rot, axes=[0,1])
    else:
        img[0] = np.flip(np.rot90(img[0], rot, axes=[0,1]), axis=1)
        img[1] = np.flip(np.rot90(img[1], rot, axes=[0, 1]), axis=1)
    return img

class bossbase_Dataset(Dataset):
    def __init__(self, cover_dir, stego_dir, sample_file):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        cover_list = sorted(glob(cover_dir + '/*'))
        stego_list = sorted(glob(stego_dir + '/*'))
        with open(sample_file) as f:
            data_name = f.readlines()
            self.data_list = [a[3:-1].strip() for a in data_name]
            self.data_num = len(self.data_list)

    def __len__(self):
        return self.data_num

    def __getitem__(self, item):
        img = np.zeros((2,256,256), dtype=float)
        label = np.array([0,1])

        img_cover = sio.loadmat(self.cover_dir + self.data_list[item])
        img_stego = sio.loadmat(self.stego_dir + self.data_list[item])
        # print(np.sum(img_cover['im']==img_stego['im']))
        img[0] = img_cover['im']
        img[1] = img_stego['im']
        img = rot_flip_pair(img)
        img_data = torch.FloatTensor(img)
        label = torch.LongTensor(label)
        return img_data, label

def return_dataloader(cover_dir, stego_dir, sample_file, type, batch_size):
    print('Loading ' + type + ' data')
    print('.......')
    a_dataset = bossbase_Dataset(cover_dir,stego_dir,sample_file)
    a_dataloader = DataLoader(a_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    a_len = 2*a_dataset.__len__()
    print("The number of "+type+" is {}".format(a_len))
    print('end')
    return a_dataloader, a_len




if __name__=='__main__':
    cover_dir = cfg['cover_dir']
    stego_dir = cfg['stego_dir']
    train_file = cfg['train_file']
    valid_file = cfg['valid_file']
    test_file = cfg['test_file']

    a, a_len = return_dataloader(cover_dir, stego_dir, train_file, 'train', 5)
    print(a_len)
    for index, data in enumerate(a):
        inputs, labels = data
