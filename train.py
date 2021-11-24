"""Source code được tham khảo từ RIS AI HairNet Hair & Head Segmentation"""

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.models import Model, load_model
from keras.layers.merge import concatenate, add

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x


def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    p5 = MaxPooling2D((2, 2))(c5)
    p5 = Dropout(dropout)(p5)

    c6 = conv2d_block(p5, n_filters = n_filters * 32, kernel_size = 3, batchnorm = batchnorm)
    # Expansive Path
    u7 = Conv2DTranspose(n_filters * 16, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c5])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c4])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c3])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u10 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c9)
    u10 = concatenate([u10, c2])
    u10 = Dropout(dropout)(u10)
    c10 = conv2d_block(u10, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

    u11 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c10)
    u11 = concatenate([u11, c1])
    u11 = Dropout(dropout)(u11)
    c11 = conv2d_block(u11, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c11)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
im_height = 224
im_width = 224
input_img = Input((im_height, im_width, 3), name = 'img')
mainmod=get_unet(input_img)
print(mainmod.summary())

import glob
import cv2
import tensorflow as tf
import tempfile
import os
from tensorflow.python.keras import optimizers
import numpy as np
import matplotlib as plt
from skimage.util.dtype import convert
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input

number_channel = 3
im_width = 224
im_height = 224
border = 5

def loading_data(path_img_train, path_mask_train):
    #This function load training data (faces and mask, 128x128)
    ids = next(os.walk(path_img_train))[2] # list of names all images in the given path
    print("No. of train images = ", len(ids))
    X = np.zeros((len(ids), im_height, im_width, number_channel), dtype=np.float32)
    y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    
    for n in range(len(ids)):
      img = load_img(path_img_train+'/'+ids[n], grayscale=False)
      x_img = img_to_array(img)
      x_img = resize(x_img, (im_height, im_width, number_channel), mode = 'constant', preserve_range = True)
      # Load masks
      mask = img_to_array(load_img(path_mask_train+'/'+ids[n], grayscale=True))
      mask = resize(mask, (im_width, im_width, 1), mode = 'constant', preserve_range = True)
      # Save images
      X[n] = x_img/255.0
      y[n] = mask/255.0
        
    print('Data Loaded')
    
    return X, y
path_img_train = '/content/drive/MyDrive/HK5/XLA/HairSegmentation/dataset/faces_train'
path_mask_train = '/content/drive/MyDrive/HK5/XLA/HairSegmentation/dataset/mask_train'

X, y = loading_data(path_img_train, path_mask_train)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

input_img = Input((im_height, im_width, 3), name = 'img')

model = get_unet(input_img)
print(model.summary)

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
tensorboard= TensorBoard(log_dir='logs',
                        histogram_freq=0,
                        write_graph=True,
                        write_images=True,
                        write_grads=True,
                        update_freq='epoch')

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('/content/drive/MyDrive/HK5/XLA/HairSegmentation/weight.h5', verbose=1, save_best_only=True, save_weights_only=True),
    tensorboard
]

model.fit(X_train, y_train, batch_size=5, epochs=50, callbacks=callbacks, validation_data=(X_valid, y_valid))

MODEL_DIR = tempfile.gettempdir()
version=5
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))
if os.path.isdir(export_path):
    print('\nAlready save a model, cleaning up\n')
    import shutil
    shutil.rmtree(export_path)

tf.saved_model.save(model, export_path)
print('\nSaved model:')
converter = tf.lite.TFLiteConverter.from_saved_model(export_path)
tflite_model = converter.convert()
open("head_V4_mocel_.tflite", "wb").write(tflite_model)