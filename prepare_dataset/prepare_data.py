import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage as ndi
from scipy import signal
import pickle
from sklearn.preprocessing import OneHotEncoder
def crop_nodule(coord,image,dim=112):
    """
    Returns a cropped image of size 2*dim x 2*dim when dim. There are corner cases in the border.
    We pad the image if the coordinate is located on the corner
    
    Args:
    coord: coordinate of x,y as list or tuple. For COM the input is as (y coord, x coord)
    image: image to be cropped
    dim: 1/2 size of the image after being cropped
    
    Returns:
    Cropped Image
    
    """
    x_coord = int(coord[1])+ dim
    y_coord = int(coord[0])+ dim
    
    # pad the image
    image_pad = np.pad(image, ((dim,dim),(dim,dim)), 'constant', constant_values=0)

    return image_pad[y_coord-dim:y_coord+dim,x_coord-dim:x_coord+dim]

def gradient_transform(patch):
    xder = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    yder = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    arrx = signal.convolve2d(patch,xder,mode='same')
    arry = signal.convolve2d(patch,yder,mode='same')

    return np.hypot(arrx,arry)

def laplacian_transform(patch):
    xder = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    yder = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    arrx = signal.convolve2d(patch,xder,mode='same')
    arry = signal.convolve2d(patch,yder,mode='same')
    arrx = signal.convolve2d(arrx,yder,mode='same')
    arry = signal.convolve2d(arry,xder,mode='same')
    return np.hypot(arrx,arry)


# Lets try with Mask layer later.
def create_3channel(coord,image):
    patch = crop_nodule(coord,image,dim=112)
    grad_patch = gradient_transform(patch)
    lap_patch = laplacian_transform(patch)
    output = np.stack([patch,grad_patch,lap_patch],axis=2)
    return output
    
def main():
    # Directory to load data with nodules
    IMAGE_DIR = '/home/LUNG_DATA/Image_1/'
    MASK_DIR = '/home/LUNG_DATA/Mask_1/'

    # Directory to load data without any nodules, Thus, a clean lung image
    CLEAN_DIR_IMG ='/home/LUNG_DATA/Clean_1/Image/'

    # Directory to save data
    train_output_rgb_dir = '/home/LUNG_DATA/Efficient_net/train/'
    test_output_rgb_dir = '/home/LUNG_DATA/Efficient_net/test/'
    data_label ='/home/LUNG_DATA/Efficient_net/label/'


    #Meta Information
    meta = pd.read_csv('/home/LUNG_DATA/meta_csv/meta.csv')

    #Clean Meta Information
    clean_meta = pd.read_csv('/home/LUNG_DATA/meta_csv/clean_meta.csv')


    # Get train/test label from meta.csv
    meta['original_image']= meta['original_image'].apply(lambda x:IMAGE_DIR + x +'.npy')
    meta['mask_image'] = meta['mask_image'].apply(lambda x:MASK_DIR + x +'.npy')

    # Get train/test label from meta.csv
    clean_meta['original_image']= clean_meta['original_image'].apply(lambda x:CLEAN_DIR_IMG + x +'.npy')

    # Get images that were used to train Segmentation model and that is also not labeled as Ambiguous
    train_meta = meta[(meta['data_split']!='Test') & (meta['is_cancer']!='Ambiguous')]
    test_meta = meta[(meta['data_split']=='Test') & (meta['is_cancer']!='Ambiguous')]
    train_image_paths = list(train_meta['original_image'])
    train_mask_paths = list(train_meta['mask_image'])
    train_label = list(train_meta['is_cancer'].apply(lambda x: 1 if x=='True' else 0))

    test_image_paths = list(test_meta['original_image'])
    test_mask_paths = list(test_meta['mask_image'])
    test_label = list(test_meta['is_cancer'].apply(lambda x: 1 if x=='True' else 0))

    # Get clean images directory as list
    # Only get 20% of train_image_paths
    """proportion_train = int(len(train_meta)*0.2)
    proportion_test = int(len(test_meta)*0.2)
    clean_images = list(clean_meta['original_image'])
    train_clean_paths = clean_images[:proportion_train]
    test_clean_paths = clean_images[proportion_train:proportion_train+proportion_test]
    clean_label_train = [0 for x in range(len(train_clean_paths))]
    clean_label_test = [0 for x in range(len(test_clean_paths))]"""


    # Extend the original train, test list
    # DOO  ITTT FROMMM HEREEE


    # This will ensure that only the train/validation images that were used in U-Net will be used to train the cancer classifier
    
    print("*"*50)
    print("The length of image are train: {} test: {}".format(len(train_image_paths),len(test_image_paths)))
    print("Ambiguity will be filtered")
    #load label file
    for train_img,train_mask in zip(train_image_paths,train_mask_paths):
        naming = train_img[-23:]
        mask = np.load(train_mask)
        image = np.load(train_img)
        rgb= create_3channel(ndi.center_of_mass(mask),image)
        print("Saved {}".format(naming))
        np.save(train_output_rgb_dir+naming,rgb)
    
    for test_img,test_mask in zip(test_image_paths,test_mask_paths):
        naming = test_img[-23:]
        mask = np.load(test_mask)
        image = np.load(test_img)
        rgb= create_3channel(ndi.center_of_mass(mask),image)
        print("Saved {}".format(naming))
        np.save(test_output_rgb_dir+naming,rgb)

    with open(data_label+'train.txt','wb') as fp:
        pickle.dump(train_label,fp)
    with open(data_label+'test.txt','wb') as fp:
        pickle.dump(test_label,fp)      
    print("TOTAL OF CANCER: {}, NON-CANCEROUS:{}, CLEAN:{} IMAGES WERE SAVED FOR TRAIN".format(np.sum(train_label),len(train_label)-np.sum(train_label)))
    print("TOTAL OF {} CANCER and {} NON-CANCER IMAGES WERE SAVED FOR TEST".format(np.sum(test_label),len(train_label)-np.sum(test_label)))

if __name__ == "__main__":
    main()

