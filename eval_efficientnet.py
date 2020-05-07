import pandas as pd
import numpy as np
import argparse
import os
from collections import OrderedDict
from glob import glob
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml

import torch
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import datasets, models, transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet

from metric import Confusion_matrix2
from utils import *
from classifier_dataset import ClassifierDataset


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_version', default=0,type= int,
                        help='model name: efficientnetb- ',choices=[0,1,2,3,4,5,6])
    # data
    parser.add_argument('--augmentation',default=False)

    args = parser.parse_args()

    return args

def evaluate(model,test_loader):


    with torch.no_grad():

        counter = 0
        pbar = tqdm(total=len(test_loader))
        TP,TN,FN,FP = 0,0,0,0
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()

            output = model(images)
            outputs = output.view(-1)
            labels = labels.type_as(outputs)
            TP_batch,TN_batch,FN_batch,FP_batch = Confusion_matrix2(outputs,labels)
            TP += TP_batch
            TN += TN_batch
            FN += FN_batch
            FP += FP_batch


            pbar.update(1)
        pbar.close()

        accuracy = (TP+TN)/ (TP +TN +FN +FP)
        sensitivity = TP / (TP+FN)
        specificity = TN / (TN+FP)
    torch.cuda.empty_cache()

    return accuracy, sensitivity, specificity, TP,TN,FN,FP



def main():
    args = vars(parse_args())

    print(args)

    OUTPUT_DIR = '/home/jaeho_ubuntu/Classification/model_output/efficientnetb{}/'.format(args['model_version'])
    TEST_DIR = '/home/LUNG_DATA/Efficient_net/test/'
    CLEAN_TEST_DIR = '/home/LUNG_DATA/Efficient_net/clean_test/'
    LABEL_DIR ='/home/LUNG_DATA/Efficient_net/label/'

    with open(OUTPUT_DIR+'config.yml', 'r') as f:
        config = yaml.load(f)
    with open(LABEL_DIR+'test.txt','rb') as fp:
        test_label = pickle.load(fp)
    with open(LABEL_DIR+'clean_test.txt','rb') as fp:
        clean_test_label = pickle.load(fp)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True
    model = EfficientNet.from_pretrained('efficientnet-b{}'.format(args['model_version']))

    #Fine tuning top layers
    num_ftrs = model._fc.in_features
    model._fc = nn.Sequential(nn.Linear(num_ftrs,1),
                                nn.Sigmoid())


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

    model.load_state_dict(torch.load(OUTPUT_DIR +'model.pth'))
    model = model.cuda()

    # Get image files
    test_image_paths = load_directories(TEST_DIR)
    clean_test_image_paths = load_directories(CLEAN_TEST_DIR)

    test_image_paths.extend(clean_test_image_paths)
    test_label.extend(clean_test_label)



    assert len(test_image_paths)==len(test_label), "Length of test images and test label not same"
    print("============================TESTING===========================================")
    print("Cancer nodules:{} Non Cancer nodules:{}".format(np.sum(test_label),len(test_label)-np.sum(test_label)))
    print("Ratio is {:4f}".format(np.sum(test_label)/(len(test_label)-np.sum(test_label))))

    test_dataset = ClassifierDataset(test_image_paths,test_label,Albumentation=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=6)

    log= pd.DataFrame(index=[],columns= ['test_size','accuracy','sensitivity','specificity','TP','TN','FN','FP'])

    accuracy, sensitivity, specificity, TP, TN, FN, FP = evaluate(model,test_loader)

    tmp = pd.Series([
        len(test_dataset),accuracy, sensitivity, specificity, TP, TN, FN, FP
    ], index=['test_size','accuracy','sensitivity','specificity','TP','TN','FN','FP'])

    print('Test accuracy:{:.4f}, Test sensitivity:{:.4f}, Testspecificity:{:.4f}'.format(accuracy,sensitivity,specificity))

    log = log.append(tmp,ignore_index=True)
    log.to_csv(OUTPUT_DIR +'test_result.csv',index=False)
    print("OUTPUT RESULT SAVED AS CSV in", OUTPUT_DIR)
if __name__ == '__main__':
    main()