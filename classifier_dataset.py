import os
import numpy as np
import glob
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
import torchvision
from torchvision import transforms

import albumentations as albu
from albumentations.pytorch import ToTensorV2
from albumentations import OneOf,Compose


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
            albu.Normalize(),
            OneOf([albu.HorizontalFlip(),
                   albu.VerticalFlip(),
                   albu.RandomRotate90(),
                   ],p=0.2),
            albu.ElasticTransform(alpha=1.1,alpha_affine=0.5,sigma=5,p=0.15),
            ToTensorV2()
        ])
        self.transformations = transforms.Compose([
            #transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def transform(self, image):
        #Transform to tensor
        if self.albumentation:
            #It is always best to convert the make input to 3 dimensional for albumentation
            augmented=  self.albu_transformations(image=image)
            image = augmented['image']
        else:
            image = self.transformations(image)
        image= image.type(torch.FloatTensor),
        return image

    def __getitem__(self, index):
        image = np.load(self.image_paths[index])
        image = self.transform(image)
        return image[0],self.labels[index]

    def __len__(self):
        return len(self.image_paths)
