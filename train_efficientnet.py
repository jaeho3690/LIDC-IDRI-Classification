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

from metric import Confusion_matrix
from utils import *
from classifier_dataset import ClassifierDataset


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_version', default=0,type=int,
                        help='model name: efficientnetb- ',choices=[0,1,2,3,4,5,6])
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=24, type=int,
                        metavar='N', help='mini-batch size (default: 24)')
    parser.add_argument('--early_stopping', default=30, type=int,
                        metavar='N', help='early stopping (default: 30)')
    parser.add_argument('--num_workers', default=4, type=int)

    # data
    parser.add_argument('--augmentation',default=False)
    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-6, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    config = parser.parse_args()

    return config


def train(train_loader,model,criterion,optimizer):
    avg_meters = {'loss': AverageMeter(),
                'accuracy': AverageMeter(),
                'sensitivity':AverageMeter(),
                'specificity': AverageMeter()}

    model.train()
    pbar = tqdm(total=len(train_loader))
    for images, labels in train_loader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        outputs = outputs.view(-1)
        labels = labels.type_as(outputs)
        loss = criterion(outputs, labels)
        accuracy, sensitivity, specificity = Confusion_matrix(outputs,labels)
        #print(loss)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(),images.size(0))
        avg_meters['accuracy'].update(accuracy,images.size(0))
        avg_meters['sensitivity'].update(sensitivity,images.size(0))
        avg_meters['specificity'].update(specificity,images.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('accuracy', avg_meters['accuracy'].avg),
            ('sensitivity', avg_meters['sensitivity'].avg),
            ('specificity', avg_meters['specificity'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('accuracy', avg_meters['accuracy'].avg),
                        ('sensitivity', avg_meters['sensitivity'].avg),
                        ('specificity', avg_meters['specificity'].avg),])

def validate(val_loader,model,criterion):
    avg_meters = {'loss': AverageMeter(),
                'accuracy': AverageMeter(),
                'sensitivity':AverageMeter(),
                'specificity': AverageMeter()}

    model.eval()

    pbar = tqdm(total=len(val_loader))
    for images, labels in val_loader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        outputs = outputs.view(-1)
        labels = labels.type_as(outputs)
        loss = criterion(outputs, labels)
        accuracy, sensitivity, specificity = Confusion_matrix(outputs,labels)

        avg_meters['loss'].update(loss.item(),images.size(0))
        avg_meters['accuracy'].update(accuracy,images.size(0))
        avg_meters['sensitivity'].update(sensitivity,images.size(0))
        avg_meters['specificity'].update(specificity,images.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('accuracy', avg_meters['accuracy'].avg),
            ('sensitivity', avg_meters['sensitivity'].avg),
            ('specificity', avg_meters['specificity'].avg)
        ])

        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()


    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('accuracy', avg_meters['accuracy'].avg),
                        ('sensitivity', avg_meters['sensitivity'].avg),
                        ('specificity', avg_meters['specificity'].avg),])




def main():
    # Get configuration
    config = vars(parse_args())

    # Model Output directory
    OUTPUT_DIR = '/home/jaeho_ubuntu/Classification/model_output/'
    os.makedirs(OUTPUT_DIR+'efficientnetb{}'.format(config['model_version']),exist_ok=True)
    print('Made directory called efficientnetb{}'.format(config['model_version']))

    print('-' * 20)
    print("Configuration Setting as follow")
    for key in config:
        print('{}: {}'.format(key, config[key]))
    print('-' * 20)
    
    #save configuration
    with open(OUTPUT_DIR+'efficientnetb{}/config.yml'.format(config['model_version']), 'w') as f:
        yaml.dump(config, f)

    # Data directory
    TRAIN_DIR = '/home/LUNG_DATA/Efficient_net/train/'
    VAL_DIR = '/home/LUNG_DATA/Efficient_net/val/'
    CLEAN_TRAIN_DIR ='/home/LUNG_DATA/Efficient_net/clean_train/'
    CLEAN_VAL_DIR = '/home/LUNG_DATA/Efficient_net/clean_val/'
    LABEL_DIR = '/home/LUNG_DATA/Efficient_net/label/'



    with open(LABEL_DIR+'train.txt','rb') as fp:
        train_label = pickle.load(fp)
    with open(LABEL_DIR+'val.txt','rb') as fp:
        val_label = pickle.load(fp)
    with open(LABEL_DIR+'clean_train.txt','rb') as fp:
        clean_train_label = pickle.load(fp)
    with open(LABEL_DIR+'clean_val.txt','rb') as fp:
        clean_val_label = pickle.load(fp)


    # Get image files path as list
    train_image_paths = load_directories(TRAIN_DIR)
    val_image_paths = load_directories(VAL_DIR)
    clean_train_images_paths = load_directories(CLEAN_TRAIN_DIR)
    clean_val_images_paths = load_directories(CLEAN_VAL_DIR)

    train_image_paths.extend(clean_train_images_paths)
    val_image_paths.extend(clean_val_images_paths)
    train_label.extend(clean_train_label)
    val_label.extend(clean_val_label)


    print("="*50)
    print("The length of image are train: {} validation: {}".format(len(train_image_paths),len(val_image_paths)))

    print("============================TRAINING===========================================")
    print("Cancer nodules:{} Non Cancer nodules:{}".format(np.sum(train_label),len(train_label)-np.sum(train_label)))
    print("Ratio is {:4f}".format(np.sum(train_label)/(len(train_label)-np.sum(train_label))))
    print("============================VALIDATION=========================================")
    print("Cancer nodules:{} Non Cancer nodules:{}".format(np.sum(val_label),len(val_label)-np.sum(val_label)))
    print("Ratio is {:4f}".format(np.sum(val_label)/(len(val_label)-np.sum(val_label))))
    # Create Dataset
    train_dataset = ClassifierDataset(train_image_paths,train_label,config['augmentation'])
    val_dataset = ClassifierDataset(val_image_paths,val_label,config['augmentation'])
    
    # Model
    cudnn.benchmark = True
    model = EfficientNet.from_pretrained('efficientnet-b{}'.format(config['model_version']))

    #Fine tuning top layers
    num_ftrs = model._fc.in_features
    model._fc = nn.Sequential(nn.Linear(num_ftrs,1),
                                nn.Sigmoid())

    criterion = nn.BCEWithLogitsLoss().cuda()

    # Decay LR by a factor of 0.1 every 7 epochs

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.cuda()
    params = filter(lambda p: p.requires_grad, model.parameters())

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # Create Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=6)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=6)
    
    log= pd.DataFrame(index=[],columns= ['epoch', 'loss', 'accuracy','sensitivity','specificity,',
                                        'val_loss', 'val_accuracy','val_sensitivity','val_specificity'])

    best_loss= 10
    trigger = 0

    for epoch in range(config['epochs']):

        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion)

        print('Training epoch [{}/{}], Training BCE loss:{:.4f}, Training accuracy:{:.4f}, Training sensitivity:{:.4f}, Training specificity:{:.4f}, \
                    Validation BCE loss:{:.4f}, Validation accuracy:{:.4f}, Validation sensitivity:{:.4f}, Validation specificity:{:.4f},'.format( 
            epoch + 1, config['epochs'], train_log['loss'], train_log['accuracy'], train_log['sensitivity'], train_log['specificity'],val_log['loss'], val_log['accuracy'],val_log['sensitivity'], val_log['specificity']))

        tmp = pd.Series([
            epoch,
            train_log['loss'],
            train_log['accuracy'],
            train_log['sensitivity'],
            train_log['specificity'],
            val_log['loss'],
            val_log['accuracy'],
            val_log['sensitivity'],
            val_log['specificity'],
        ], index=['epoch', 'loss', 'accuracy','sensitivity','specificity,','val_loss', 'val_accuracy','val_sensitivity','val_specificity'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv(OUTPUT_DIR+'efficientnetb{}/log.csv'.format(config['model_version']), index=False)

        trigger += 1

        if val_log['loss'] < best_loss:
            torch.save(model.state_dict(), OUTPUT_DIR+'efficientnetb{}/model.pth'.format(config['model_version']))
            best_loss= val_log['loss']
            print("=> saved best model as validation loss is greater than previous best loss")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()


    
