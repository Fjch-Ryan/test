import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate

class ISICDataset(Dataset):
    def __init__(self, args, data_path , transform = None, mode = 'Training',plane = False):


        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()    #loc函数：通过行索引 “Index” 中的具体值来取行数据（如取"Index"为"A"的行,loc[A]）
                                                  # iloc函数：通过行号来取行数据（如取第二行的数据,iloc[1]）
                                                 #注：loc是location的意思，iloc中的i是integer的意思，仅接受整数作为参数
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)
        # 在数字图像处理中，针对不同的图像格式有其特定的处理算法.
        # 对于彩色图像，不管其图像格式是PNG，还是BMP，或者JPG，在PIL中，使用Image模块的open()函数打开后，返回的图像对象的模式都是“RGB”。
        # #而对于灰度图像，不管其图像格式是PNG，还是BMP，或者JPG，打开后，其模式为“L”。
        # 对于PNG、BMP和JPG彩色图像格式之间的互相转换都可以通过Image模块的open()和save()函数来完成。具体说就是，在打开这些图像时，PIL会将它们解码为三通道的“RGB”图像。用户可以基于这个“RGB”图像，
        # 对其进行处理。处理完毕，使用函数save()，可以将处理结果保存成PNG、BMP和JPG中任何格式。
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)


        return (img, mask, name)