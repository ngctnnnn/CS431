import os
import numpy as np
import pandas as pd
from skimage import io, exposure

def make_lungs():
    path = 'JSRT/raw/'
    for i, filename in enumerate(os.listdir(path)):
        img = 1.0 - np.fromfile(path + filename, dtype='>u2').reshape((2048, 2048)) * 1. / 4096
        img = exposure.equalize_hist(img)
        io.imsave('JSRT/preprocessed/' + filename[:-4] + '.png', img)
        print('Lung', i, filename)

def make_masks():
    path = 'MC/raw/ManualMask/leftMask'
    for i, filename in enumerate(os.listdir(path)):
        left = io.imread('MC/raw/ManualMask/leftMask/' + filename[:-4] + '.png')
        right = io.imread('MC/raw/ManualMask/rightMask/' + filename[:-4] + '.png')
        io.imsave('MC/preprocessed/masks/' + filename[:-4] + '_msk.png', np.clip(left + right, 0, 255))
        # print('Mask', i, filename)

def split_cancer_healthy():
    """
    JPCLN -- Nodule + lung cancer
    JPCNN -- Non-nodule
    """
    
    path = 'JSRT/preprocessed/data/' 
    for i, filename in enumerate(os.listdir(path)):
        if filename[:-7] == "JPCLN":
            os.system('mv ~/Desktop/uni-study/CS431/final-project/JSRT/preprocessed/' + filename + ' ~/Desktop/uni-study/CS431/final-project/JSRT/preprocessed/Nodule/')
        if filename[:-7] == "JPCNN":
            os.system('mv ~/Desktop/uni-study/CS431/final-project/JSRT/preprocessed/' + filename + ' ~/Desktop/uni-study/CS431/final-project/JSRT/preprocessed/Non-nodule/')

def split_malignant_benign():
    """
    malignant -- U ác tính
    benign -- U lành tính
    """
    # Query: df[df['image_id'] == 'JPCLN001.IMG']['type_of']
    
    path = 'JSRT/preprocessed/Nodule/'
    df = pd.read_csv('~/Desktop/uni-study/CS431/final-project/demonstration/CLN_demonstration.csv')
    malignant_list = list(df[df['type_of']=='malignant']['image_id'])
    benign_list = list(df[df['type_of']=='benign']['image_id'])
    for i, filename in enumerate(os.listdir(path)):
        if filename[:-4] + '.IMG' in malignant_list:
           os.system('mv ~/Desktop/uni-study/CS431/final-project/JSRT/preprocessed/Nodule/' + filename + ' ~/Desktop/uni-study/CS431/final-project/JSRT/preprocessed/Nodule/Cancer/')  
        if filename[:-4] + '.IMG' in benign_list:
            os.system('mv ~/Desktop/uni-study/CS431/final-project/JSRT/preprocessed/Nodule/' + filename + ' ~/Desktop/uni-study/CS431/final-project/JSRT/preprocessed/Nodule/Benign/') 

def process_MC_CXR():
    path = 'MC/raw/CXR_png/'
    os.system('mv ' + path + '*.png' + ' ~/Desktop/uni-study/CS431/final-project/MC/preprocessed/CXR/')
    print(class_name[2])

def main():
    # make_lungs()
    # make_masks()
    # split_cancer_healthy()
    # split_malignant_benign()
    process_MC_CXR()
 
if __name__=="__main__":
    main()