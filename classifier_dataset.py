import os
import numpy as np
import glob

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
import torchvision
from torchvision import transforms

import albumentations as albu
from albumentations.pytorch import ToTensorV2

class ClassifierDataset(Dataset):
    def __init__(self, IMAGES_PATHS,label,Albumentation=False):
        """
        IMAGES_PATHS: list of images paths ['./Images/0001_01_images.npy','./Images/0001_02_images.npy']
        label: list of labels
        Albumentation: Whether to apply augmentation
        """
        self.image_paths = IMAGES_PATHS
        self.labels = label
        self.albumentation = Albumentation

        self.albu_transformations =  albu.Compose([
            albu.ElasticTransform(alpha=1.1,alpha_affine=0.5,sigma=5,p=0.15),
            albu.HorizontalFlip(p=0.15),
            ToTensorV2()
        ])
        self.transformations = transforms.Compose([transforms.ToTensor()])

    def transform(self, image):
        #Transform to tensor
        if self.albumentation:
            #It is always best to convert the make input to 3 dimensional for albumentation
            image = image.reshape(512,512,1)
            augmented=  self.albu_transformations(image=image)
            image = augmented['image']
        else:
            image = self.transformations(image)
        image= image.type(torch.FloatTensor), 
        return image

    def __getitem__(self, index):
        image = np.load(self.image_paths[index])
        image= self.transform(image)
        return image,self.labels[index]

    def __len__(self):
        return len(self.image_paths)
