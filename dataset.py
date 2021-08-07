import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import re
import sys
import random
from torchvision import transforms


# rainy_dataset中的每一条数据，都是一个形如：
# (ground_img, rainy_img1, rainy_img2, ...)
# 的tuple
# 调用__getitem__时，会返回(tuple_len,data_sample)
# tuple_len为data_sample的长度，data_sample即形如(ground_img, rainy_img1, rainy_img2, ...)
DEFAULT_data_path="toy_datasets"
DEFAULT_data_subpath="training"    # 如果是用来test则选择"testing"文件夹
DEFAULT_ground_path="ground_truth"
DEFAULT_rainy_path="rainy_image"
DEFAULT_format="jpg"
DEFAULT_image_size=128


class rainy_dataset(Dataset):
    # data_len=-1则表示选取"ground_truth"文件夹中的所有图像
    # rainy_data_len表示选取多少个对应的rainy的图像
    # eg:若rainy_data_len=3，则“52.jpg”对应到的rainy_image为"52_1.jpg,52_2.jpg,52_3.jpg""
    def __init__(self, data_path=DEFAULT_data_path, data_subpath=DEFAULT_data_subpath, data_len=-1, rainy_data_len=1, normalize = True):
        self.path = data_path+"/"+data_subpath
        self.image_labels = []
        self.rainy_range = rainy_data_len
        self.normalize=normalize

        all_img_labels=[]
        ground_path=self.path+"/"+DEFAULT_ground_path
        form=re.compile("\S+\."+DEFAULT_format)
        for filename in os.listdir(ground_path):
            if form.match(filename):
                filename=re.sub("\."+DEFAULT_format,"",filename)
                all_img_labels.append(filename)

        if data_len==-1:
            self.image_labels=all_img_labels
        else:
            try:
                if len(all_img_labels) >= data_len:
                    self.image_labels=random.sample(all_img_labels,data_len)
                else:
                    raise Exception("data_len too large!")
            except Exception as exp:
                for string in exp.args:
                    print(string)
                sys.exit()

        try:
            sample_label=self.image_labels[0]
            rainy_path=self.path+"/"+DEFAULT_rainy_path
            form=re.compile(sample_label+"_"+"\S+\."+DEFAULT_format)
            count=0
            for filename in os.listdir(rainy_path):
                if form.match(filename):
                    count+=1
            if count < rainy_data_len:
                raise Exception("rainy_data_len too large!")
        except Exception as exp:
            for string in exp.args:
                print(string)
            sys.exit()


    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        label=self.image_labels[index]
        path_ground=self.path+"/"+DEFAULT_ground_path+"/"+label+"."+DEFAULT_format
        path_rainy=[]
        for rainy_label in range(1,self.rainy_range+1):
            path_rainy.append(self.path+"/"+DEFAULT_rainy_path+"/"+label+"_"+str(rainy_label)+"."+DEFAULT_format)

        tuple_len = self.rainy_range + 1
        tuple_self = []
        tuple_self.append(Image.open(path_ground))
        for a_path_rainy in path_rainy:
            tuple_self.append(Image.open(a_path_rainy))
        # crop and reshape
        for i in range(tuple_len):
            im = tuple_self[i]
            width, height = im.size
            left=0
            top=0
            if width > height :
                bottom=height
                right=height
            else:
                bottom=width
                right=width
            im = im.crop((left, top, right, bottom))
            im = im.resize((DEFAULT_image_size,DEFAULT_image_size))
            tuple_self[i]=im

        if self.normalize:
            trans = transforms.Compose([transforms.ToTensor()])
            for i in range(len(tuple_self)):
                tuple_self[i]=(trans(tuple_self[i])-0.5)/0.5
        tuple_self=tuple(tuple_self)
        return tuple_len, tuple_self

