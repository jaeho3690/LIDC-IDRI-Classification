import pandas as pd
import argparse
import os
from collections import OrderedDict
from glob import glob

from sklearn.model_selection import train_test_split

from utils import get_test_label
from classifier_dataset import ClassifierDataset

import torch
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image


from efficientnet_pytorch import EfficientNet


def main():

    IMAGE_DIR = '/home/LUNG_DATA/Image'
    CLEAN_DIR_IMG ='/home/LUNG_DATA/Clean/Image'
    meta = '/home/LUNG_DATA/meta_info.csv'
    test_size = 0.2
    validation_proportion = 0.25

    # Get image files
    folder_images = list()
    for (dirpath, _ , filenames) in os.walk(IMAGE_DIR):
        folder_images += [os.path.join(dirpath, file) for file in filenames]
    for (dirpath, _ , filenames) in os.walk(CLEAN_DIR_IMG):
        folder_clean_images += [os.path.join(dirpath, file) for file in filenames]
    folder_images.sort()
    folder_clean_images.sort()

    # This will ensure that only the train/validation images that were used in U-Net will be used to train the cancer classifier
    train_image_paths,test_image_paths,train_clean_images_paths,test_clean_images_paths = train_test_split(folder_images,folder_clean_images,test_size=test_size,random_state=1)
    train_image_paths.extend(train_clean_images_paths)
    test_image_paths.extend(test_clean_images_paths)
    print("*"*50)
    print("The length of image are train: {} test: {}".format(len(train_image_paths),len(test_image_paths)))
    print("Ambiguity will be filtered")
    #load label file
    meta = pd.read_csv(meta)

    train_X, train_y = get_test_label(meta,train_image_paths)
    val_X, val_y = train_test_split(train_X,train_y,validation_proportion)
    test_X, test_y = get_test_label(meta,test_image_paths)

    print("*"*50)
    print("Train: {}  Validation: {}  Test: {}".format(len(train_X),len(val_X),len(test_X)))

    # Create Dataset
    train_dataset = ClassifierDataset(train_X,train_y,config['augmentation'])
    val_dataset = ClassifierDataset(val_X,val_y,config['augmentation'])



    
