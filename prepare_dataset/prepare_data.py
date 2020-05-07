import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage as ndi
from scipy import signal
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import image


def get_path_label(meta,is_clean=False):
    """This returns the path of image and mask and label list """
    image_path = list(meta['original_image'])
    mask_path = list(meta['mask_image'])
    if is_clean:
        label = [0 for x in range(len(image_path))]
    else:
        label = list(meta['is_cancer'].apply(lambda x: 1 if x=='True' else 0))
    return image_path,mask_path,label

def crop_clean_patch(clean_image,dim=224):
    """Crop random patch size of dim from clean dataset"""
    clean_image = clean_image[100:400,100:400]
    for i in range(100):  
        patch = image.extract_patches_2d(clean_image,(dim,dim),1)
        patch = np.squeeze(patch)
        if np.sum(patch)> 2000:
            return patch
    return patch


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
# Lets try with Mask layer later.
def create_3channel_clean(image):
    """create 3 channel for clean dataset"""
    patch = crop_clean_patch(image)
    grad_patch = gradient_transform(patch)
    lap_patch = laplacian_transform(patch)
    output = np.stack([patch,grad_patch,lap_patch],axis=2)
    return output

    
def main():
    # Directory to load data with nodules
    IMAGE_DIR = '/home/LUNG_DATA/Image_1/'
    MASK_DIR = '/home/LUNG_DATA/Mask_1/'

    # Directory to load data without any nodules, Thus, a clean lung image
    CLEAN_DIR_IMG ='/home/LUNG_DATA/Clean/Image/'


    # Directory to save data
    train_output_rgb_dir = '/home/LUNG_DATA/Efficient_net/train/'
    val_output_rgb_dir = '/home/LUNG_DATA/Efficient_net/val/'
    test_output_rgb_dir = '/home/LUNG_DATA/Efficient_net/test/'
    data_label ='/home/LUNG_DATA/Efficient_net/label/'

    # Directory to save clean data
    clean_train_output_rgb_dir = '/home/LUNG_DATA/Efficient_net/clean_train/'
    clean_val_output_rgb_dir = '/home/LUNG_DATA/Efficient_net/clean_val/'
    clean_test_output_rgb_dir = '/home/LUNG_DATA/Efficient_net/clean_test/'    

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
    train_meta = meta[(meta['data_split']=='Train') & (meta['is_cancer']!='Ambiguous')]
    val_meta = meta[(meta['data_split']=='Validation') & (meta['is_cancer']!='Ambiguous')]
    test_meta = meta[(meta['data_split']=='Test') & (meta['is_cancer']!='Ambiguous')]

    # Get clean images that were used to train Segmentation model and that is also not labeled as Ambiguous
    clean_train_meta = clean_meta[(clean_meta['data_split']=='Train')]
    clean_val_meta = clean_meta[(clean_meta['data_split']=='Validation')]
    clean_test_meta = clean_meta[(clean_meta['data_split']=='Test')]

    train_image_paths,train_mask_paths, train_label  = get_path_label(train_meta)
    val_image_paths, val_mask_paths, val_label = get_path_label(val_meta)
    test_image_paths, test_mask_paths, test_label = get_path_label(test_meta)

    clean_train_image_paths,_ ,clean_train_label  = get_path_label(clean_train_meta,True)
    clean_val_image_paths,_ ,clean_val_label  = get_path_label(clean_val_meta,True)
    clean_test_image_paths,_ ,clean_test_label  = get_path_label(clean_test_meta,True)


    print("*"*50)
    print("The length of image are train: {} test: {}".format(len(train_image_paths),len(test_image_paths)))
    print("Ambiguity will be filtered")
    #load train and save as 3 channel file
    for train_img,train_mask in zip(train_image_paths,train_mask_paths):
        naming = train_img[-23:]
        mask = np.load(train_mask)
        image = np.load(train_img)
        rgb= create_3channel(ndi.center_of_mass(mask),image)
        print("Saved {}".format(naming))
        np.save(train_output_rgb_dir+naming,rgb)

    #load validation and save as 3 channel file
    for val_img,val_mask in zip(val_image_paths,val_mask_paths):
        naming = val_img[-23:]
        mask = np.load(val_mask)
        image = np.load(val_img)
        rgb= create_3channel(ndi.center_of_mass(mask),image)
        print("Saved {}".format(naming))
        np.save(val_output_rgb_dir+naming,rgb)

    for test_img,test_mask in zip(test_image_paths,test_mask_paths):
        naming = test_img[-23:]
        mask = np.load(test_mask)
        image = np.load(test_img)
        rgb= create_3channel(ndi.center_of_mass(mask),image)
        print("Saved {}".format(naming))
        np.save(test_output_rgb_dir+naming,rgb)

    ################################################################################################################
    # The dataset is imbalanced. There are more cancers than non-cancers                                           #
    # We will balance this data by adding clean dataset to the model                                               #
    # I have prefigured out how much clean data set I need for train, validation, test. The number are 783,362,830.#
    ################################################################################################################
    for train_img in clean_train_image_paths[:783]:
        naming = train_img[-23:]
        image = np.load(train_img)
        rgb = create_3channel_clean(image)
        print("Saved {}".format(naming))
        np.save(clean_train_output_rgb_dir+naming,rgb)

    for val_img in clean_val_image_paths[:362]:
        naming = val_img[-23:]
        image = np.load(val_img)
        rgb = create_3channel_clean(image)
        print("Saved {}".format(naming))
        np.save(clean_val_output_rgb_dir+naming,rgb)

    for test_img in clean_test_image_paths[:830]:
        naming = test_img[-23:]
        image = np.load(test_img)
        rgb = create_3channel_clean(image)
        print("Saved {}".format(naming))
        np.save(clean_test_output_rgb_dir+naming,rgb)

    with open(data_label+'train.txt','wb') as fp:
        pickle.dump(train_label,fp)
    with open(data_label+'val.txt','wb') as fp:
        pickle.dump(val_label,fp)
    with open(data_label+'test.txt','wb') as fp:
        pickle.dump(test_label,fp)      

    clean_train_label = clean_train_label[:783]
    clean_val_label = clean_val_label[:362]
    clean_test_label = clean_test_label[:830]

    print(clean_train_label)
    with open(data_label+'clean_train.txt','wb') as fp:
        pickle.dump(clean_train_label,fp)
    with open(data_label+'clean_val.txt','wb') as fp:
        pickle.dump(clean_val_label,fp)
    with open(data_label+'clean_test.txt','wb') as fp:
        pickle.dump(clean_test_label,fp)      

    print("TOTAL OF CANCER: {}, NON-CANCEROUS:{} IMAGES WERE SAVED FOR TRAIN".format(np.sum(train_label),len(train_label)-np.sum(train_label)))
    print("TOTAL OF CANCER: {}, NON-CANCEROUS:{} IMAGES WERE SAVED FOR VAL".format(np.sum(val_label),len(val_label)-np.sum(val_label)))
    print("TOTAL OF CANCER: {}, NON-CANCEROUS:{} IMAGES WERE SAVED FOR TEST".format(np.sum(test_label),len(train_label)-np.sum(test_label)))

    print("AS THE DATA IS IMBALANCED, WE ADDED CLEAN IMAGES AS FOLLOWING")
    print("TRAIN: {}, VAL: {}, TEST: {}".format(len(clean_train_label),len(clean_val_label),len(clean_test_label)))

if __name__ == "__main__":
    main()

