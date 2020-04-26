import pandas as pd
import numpy as np


def get_test_label(meta,folder_list):
    """This function returns the cancer label, and final data path by going through meta.csv
    I removed the ambiguous label. This is when doctors have given a label 3.
    Args:
        meta: meta df file
        folder_list: list of directories
    Returns:
        label_list: cancer label
        final_folder_list: final directories"""
    print("Getting cancer labels...")
    
    label_list =[]
    final_folder_list =[]
    for file in folder_list:
        if len(file) ==61:
            #/home/LUNG_DATA/Image/LIDC-IDRI-0819/0819_nodule2_slice19.npy
            label = meta.loc[meta['original_image']==file[-24:-4],'is_cancer'].values[0]
        else:
            #/home/LUNG_DATA/Image/LIDC-IDRI-0819/0819_nodule20_slice19.npy
            label = meta.loc[meta['original_image']==file[-25:-4],'is_cancer'].values[0]
        if label =='Ambiguous':
            pass
        elif label =='True':
            label_list.append(1)
            final_folder_list.append(file)
        else:
            label_list.append(0)
            final_folder_list.append(file)
    return label_list, final_folder_list

