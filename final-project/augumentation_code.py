from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, sys 
import numpy as np
import pandas as pd
from skimage import io, exposure, transform,  img_as_float

def load_data_JSRT(class_name, image_shape):
    """
    Function to load JSRT data with corresponding class name (Cancer, Benign, Non-nodule)
    
    Args:
    class_name -- Class name to load data from
    image_shape
    
    Return:
    X -- CXR images from dataset
    y -- Corresponding mask CXR
    """
    if class_name == 'Non-nodule':
        path = 'JSRT/preprocessed/' + class_name + '/'
    else:
        path = 'JSRT/preprocessed/Nodule/' + class_name + '/'
    
    X = []
    for i, filename in enumerate(os.listdir(path)):
        img = io.imread(path + filename)
        img = transform.resize(img, image_shape)
        img = np.expand_dims(img, -1)
        X.append(img)
        
    return X

image_shape = (256, 256)

X_non_nodule = load_data_JSRT('Non-nodule', image_shape)
X_cancer = load_data_JSRT('Cancer', image_shape)
X_benign = load_data_JSRT('Benign', image_shape)

X_non_nodule = ImageDataGenerator(rotation_range=10,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                rescale=1.,
                                zoom_range=0.2,
                                fill_mode='nearest',
                                cval=0)

X_cancer = ImageDataGenerator(rotation_range=10,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                rescale=1.,
                                zoom_range=0.2,
                                fill_mode='nearest',
                                cval=0)
X_benign = ImageDataGenerator(rotation_range=10,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                rescale=1.,
                                zoom_range=0.2,
                                fill_mode='nearest',
                                cval=0)

