# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:21:33 2024

@author: Mohammed
"""

from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import cv2
import os


def minmaxScalar(img):
    return (img - img.min()) / (img.max() - img.min()).astype(np.float32)


def get_lblmask(img, th):
    return cv2.inRange(img, th, th)


class F3Sec(Dataset):
    label_rgb_codes = {'certain': [31, 119, 180], 'uncertain': [44, 160, 44], 'no': [255, 127, 14]}

    def __init__(self, cfg, datalist, subset,classlabel):
        self.train=[]
        self.lbls=[]
        self.labelset = cfg.dataset.labelset.split('-')
        groups=os.listdir(cfg.dataset.root)
        grps= [s for s in groups if any(label in s for label in classlabel)]
        if subset == 'train': # for train set 
            for group in grps: # take all annotators 
                if not group =='expert':
                    for i,img in enumerate(datalist):
                        
                        if os.path.exists(os.path.join(cfg.dataset.root,group,img)):
                            self.train.append(os.path.join(cfg.dataset.data_path,img))
                            self.lbls.append(os.path.join(cfg.dataset.root,group,img))
                else:
                    lst=os.listdir(os.path.join(cfg.dataset.root,group))
                    for img in lst:
                        self.train.append(os.path.join(cfg.dataset.data_path,img))
                        self.lbls.append(os.path.join(cfg.dataset.root,group,img))
        else: # for test set 
            for group in grps:
                if any(label in group for label in classlabel) : # check if the annotator in the test class 
                    if not group =='expert':
                        for i,img in enumerate(datalist):
                            
                            if os.path.exists(os.path.join(cfg.dataset.root,group,img)):
                                self.lbls.append(os.path.join(cfg.dataset.root,group,img))
                                self.train.append(os.path.join(cfg.dataset.data_path,img))
                    else:
                        train=os.listdir(os.path.join(cfg.dataset.root,group))
                        test=os.listdir('C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/olives/faultseg/expertall')
                        for img in test:
                            if not img in train:
                                self.train.append(os.path.join(cfg.dataset.data_path,img))
                                self.lbls.append(os.path.join('C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/olives/faultseg/expertall',img))
        print("h")
    def _get_singleannot_label(self, labels_path):
        label_rgb_array = np.asarray(Image.open(labels_path).convert('RGB'))
        
        ## collect all types of labels
        lblmask = 0.
        for lbl in self.labelset:
            # get the label mask
            _lblmask = get_lblmask(label_rgb_array, np.array(self.label_rgb_codes[lbl]))
            # print(_lblmask.shape, _lblmask.max(), _lblmask.sum())
            lblmask += _lblmask

        label = (lblmask == 255).astype(np.uint8)
        

        return label  
        
    def __getitem__(self, index):
        sec=np.asarray(Image.open(self.train[index]).convert("RGB"))
        lbl=self._get_singleannot_label(self.lbls[index])
        return torch.tensor(minmaxScalar(np.transpose(sec, (2, 0, 1))), dtype=torch.float32), torch.tensor(lbl, dtype=torch.long)

        
    def __len__(self):
        return len(self.train)
