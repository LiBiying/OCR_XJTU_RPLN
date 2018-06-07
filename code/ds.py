# coding=utf-8

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import os
import cv2
import numpy as np


# if max length is None, it will be automatically chosen according to training set.
def get_train_val_dataset(csv_path, data_dir, transform=None, size=(256, 48), max_length=None,
                           save_character_str = False,
                          character_str_saving_path='../all_characters_f.txt'):
    frames = pd.read_csv(csv_path)
    train_idx_list = pd.read_csv('../train_list_all.csv').iloc[:,0]
    val_idx_list = pd.read_csv('../val_list_all.csv').iloc[:,0]
    if max_length is None:
        max_len = 0
    else:
        max_len = max_length

    character_set = set()
    for label in frames.iloc[:, 1]:
        label_str = str(label)
        max_len = max(max_len, len(label_str))
        for char in label_str:
            character_set.add(char)

    character_str = ''.join(sorted([char for char in character_set]))
    character_str = '_' + character_str # Using '_' as the blank symbol
    if save_character_str:
        with open(character_str_saving_path, 'w', encoding='utf8') as ac:
            ac.write(character_str+'\n')

    char2index = {char: index  for index, char in enumerate(character_str)}
#    char2index['_'] = 0

    row_idx_list = list(range(frames.shape[0]))
#    train_idx_list, val_idx_list = train_test_split(row_idx_list, test_size=val_percentage)
    train_frames = frames.iloc[train_idx_list, :]
    # print(train_frames)
    test_frames = frames.iloc[val_idx_list, :]

    return MyDataset(train_frames, data_dir, char2index, max_len, transform=transform, size=size), \
           MyDataset(test_frames, data_dir, char2index, max_len, transform=transform, size=size), \
           character_str, char2index

def get_test_dataset(data_dir, transform=None, size=(256, 48), max_length=None):
    if max_length is None:
        max_len = 0
    else:
        max_len = max_length
    return TestDataset(data_dir, max_len, transform=transform, size=size)

class MyDataset(Dataset):
    def __init__(self, frames, data_dir, char2index, max_length, transform=None, size=(256, 48), fixed_width=True):

        self.transform = transform
        self.size = size
        self.data_dir = data_dir
        # self.character_str = character_str
        # self.char2index = char2index
        self.max_length = max_length
        self.fixed_width = fixed_width

        self.img_name_list = [str(s) for s in frames.iloc[:, 0]]
        self.label_length_list = []

        label_str_list = []
        for label in frames.iloc[:, 1]:
            label_str = str(label)
            self.label_length_list.append(len(label_str))
            label_str_list.append(label_str)

        self.label_list = [[char2index[char] for char in label_str] + [0]*(self.max_length - len(label_str)) for label_str in label_str_list]


    def __getitem__(self, index):
        img_name = self.img_name_list[index]
        img = cv2.imread(os.path.join(self.data_dir, img_name))
        if img is None:
            return self[index + 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        assert len(img.shape) == 2


        if self.size is not None:
            
            if (img.shape[1], img.shape[0]) != self.size:
                # img = cv2.resize(img, (int(img.shape[1] / img.shape[0] * self.height)), self.height)
                if self.fixed_width:
                    # if (img.shape[1]>=self.size[0]) or (img.shape[0]>=self.size[1]):
                    #     img = cv2.resize(img, self.size)
                    # else:                  
                    #     img = self.img_padding(img)
                    lwr = self.size[0]/self.size[1] # length width ratio
                    if img.shape[1]/img.shape[0] > lwr:
                        img = cv2.resize(img, (self.size[0], int(img.shape[0]*self.size[0]/img.shape[1])))
                        top = int((self.size[1] - img.shape[0])/2)
                        bottom = self.size[1] - img.shape[0] - top
                        img= cv2.copyMakeBorder(img, top, bottom, 0, 0,cv2.BORDER_CONSTANT,value=255)
                    else:
                        img = cv2.resize(img, (int(img.shape[1]*self.size[1]/img.shape[0]), self.size[1]))
                        left = int((self.size[0] - img.shape[1])/2)
                        right = self.size[0] - img.shape[1] - left
                        img= cv2.copyMakeBorder(img, 0, 0, left, right,cv2.BORDER_CONSTANT,value=255)


                else:
                    img = cv2.resize(img, (int(img.shape[1]/img.shape[0]*self.size[1]), self.size[1]))
            img = img[:, :, np.newaxis]
            


        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.label_list[index], dtype=torch.int32)
        length = torch.tensor([self.label_length_list[index]], dtype=torch.int32)


        return img, label, length
    
    def img_padding(self,img):
        std_h=self.size[1]
        std_w=self.size[0]
        model = np.zeros([std_h,std_w],dtype=np.uint8)
        he = img.shape[0]
        wi = img.shape[1]
        for i,aline in enumerate(model[int((std_h-he)/2):int(std_h-(std_h-he)/2)]):
            aline[int((std_w-wi)/2):int(std_w-(std_w-wi)/2)]=img[i][:]
        return model
         

    def __len__(self):
        return len(self.img_name_list)

class TestDataset(Dataset):
    def __init__(self, data_dir,max_length, transform=None, size=(256, 48)):

        self.transform = transform
        self.size = size
        self.data_dir = data_dir
        self.max_length = max_length
        self.img_name_list = os.listdir(data_dir)

    def __getitem__(self,index):
        img_name = self.img_name_list[index]
        img = cv2.imread(os.path.join(self.data_dir, img_name))
        if img is None:
            return self[index + 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert len(img.shape) == 2

        if self.size is not None:
#            if (img.shape[1], img.shape[0]) != self.size:
#                img = cv2.resize(img, self.size)
            if (img.shape[1], img.shape[0]) != self.size:
                
                lwr = self.size[0]/self.size[1] # length width ratio
                if img.shape[1]/img.shape[0] > lwr:
                    img = cv2.resize(img, (self.size[0], int(img.shape[0]*self.size[0]/img.shape[1])))
                    top = int((self.size[1] - img.shape[0])/2)
                    bottom = self.size[1] - img.shape[0] - top
                    img= cv2.copyMakeBorder(img, top, bottom, 0, 0,cv2.BORDER_CONSTANT,value=255)
                else:
                    img = cv2.resize(img, (int(img.shape[1]*self.size[1]/img.shape[0]), self.size[1]))
                    left = int((self.size[0] - img.shape[1])/2)
                    right = self.size[0] - img.shape[1] - left
                    img= cv2.copyMakeBorder(img, 0, 0, left, right,cv2.BORDER_CONSTANT,value=255)


                # if (img.shape[1]>=self.size[0])|(img.shape[0]>=self.size[1]):
                #     img = cv2.resize(img, self.size)
                # else:
                #     img = self.img_padding(img)

            img = img[:, :, np.newaxis]
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def img_padding(self,img):
        std_h=self.size[1]
        std_w=self.size[0]
        model = np.zeros([std_h,std_w],dtype=np.uint8)
        he = img.shape[0]
        wi = img.shape[1]
        for i,aline in enumerate(model[int((std_h-he)/2):int(std_h-(std_h-he)/2)]):
            aline[int((std_w-wi)/2):int(std_w-(std_w-wi)/2)]=img[i][:]
        return model

    def __len__(self):
        return len(self.img_name_list)
        





if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # dset = MyDataset(csv_path=r'D:\_aidasai\dataset\train.csv',
    #                  data_dir=r'D:\_aidasai\dataset\train',
    #                  transform=transform)
    #
    # dloader = DataLoader(dset, batch_size=5, num_workers=2)
    #
    # for i, (inputs, labels, lengths) in enumerate(dloader):
    #     print('label:', labels.type(torch.int32), 'length:', lengths)


