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
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
from google.colab.patches import cv2_imshow
import cv2

export_path = '/content/drive/MyDrive/HK5/XLA/HairSegmentation/dataset/tmp/5'
loaded = tf.saved_model.load(export_path)
print(list(loaded.signatures.keys()))
infer = loaded.signatures["serving_default"]

"""Multiply"""

def invert(x):
  t = x.copy()
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      for k in range(x.shape[2]):
        t[i][j][k] = 255 - x[i][j][k]
  return t

def trans(mask, color = [10, 10, 10]):
  b, g, r = color
  b = 255-b
  g = 255-g
  r = 255-r
  tmp = mask.copy()
  tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
  mask = mask/255.0
  for i in range(tmp.shape[0]):
    for j in range(tmp.shape[1]):
      tmp[i][j][0] = b
      tmp[i][j][1] = g
      tmp[i][j][2] = r
      tmp[i][j] = tmp[i][j]*mask[i][j]
  return tmp

def multiply_():
  image_path = '/content/drive/MyDrive/HK5/XLA/HairSegmentation/dataset/test/non.jpg'
  image = cv2.imread(image_path)
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
  im = cv2.resize(image, (224, 224))

  input_image = np.reshape(np.array(im, dtype=np.float32), [1, 224, 224, 3])/255

  input_image = tf.convert_to_tensor(input_image, dtype=np.float32)

  out = infer(input_image)
  out = np.array(out['conv2d_46'])

  img = image.copy()

  mask = np.reshape(np.array(out, dtype=np.float32), [224, 224])
  mask_ = mask *255.0
  mask_ = cv2.resize(mask_, (image.shape[1], image.shape[0]))
  mask_ = mask_.astype(np.uint8)
  img_ = image.copy()
  m = trans(mask_, [0, 200 , 0])
  img = invert(m)
  img = img/255.0
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      image[i][j] = (image[i][j]*(img[i][j]))
  cv2.imwrite('/content/drive/MyDrive/HK5/XLA/HairSegmentation/i3.jpg', image)
  fig, ax = plt.subplots(1,3,figsize=(12,6))
  ax[0].imshow(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))
  ax[1].imshow(cv2.cvtColor(mask_, cv2.COLOR_BGR2RGB))
  ax[2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

"""AddWeighted"""
def addweighted():
  image_path = '/content/drive/MyDrive/HK5/XLA/HairSegmentation/dataset/test_face/25680.jpg'
  image = cv2.imread(image_path)
  cv2.imwrite('/content/drive/MyDrive/HK5/XLA/HairSegmentation/o2.jpg', image)
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
  im = cv2.resize(image, (224, 224))

  input_image = np.reshape(np.array(im, dtype=np.float32), [1, 224, 224, 3])/255

  input_image = tf.convert_to_tensor(input_image, dtype=np.float32)

  out = infer(input_image)
  out = np.array(out['conv2d_46'])

  img = image.copy()

  mask = np.reshape(np.array(out, dtype=np.float32), [224, 224])
  mask_ = mask *255.0
  mask_ = cv2.resize(mask_, (image.shape[1], image.shape[0]))
  mask_ = mask_.astype(np.uint8)
  cv2_imshow(mask_)
  cv2.imwrite('/content/drive/MyDrive/HK5/XLA/HairSegmentation/m2.jpg', mask_)
  color =  [47, 248 , 255]
  color_img = np.copy(image)
  _, mask_= cv2.threshold(mask_, 150, 255, cv2.THRESH_BINARY)
  for i in range(mask_.shape[0]):
    for j in range(mask_.shape[1]):
      if mask_[i][j] == 255:
        color_img[i][j] = color
  #cv2.imwrite('/content/drive/MyDrive/HK5/XLA/HairSegmentation/addweighted.jpg', color_img)
  color_img = cv2.addWeighted(color_img, 0.3, image, 0.7, 0, color_img)
  cv2.imwrite('/content/drive/MyDrive/HK5/XLA/HairSegmentation/i2.jpg', color_img)
  cv2_imshow(color_img)

"""Thử ảnh"""

def recoloring(img, mask , color):
  image = img.copy()
  m = trans(mask, color)
  img = invert(m)
  img = img/255.0
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      image[i][j] = (image[i][j]*(img[i][j]))
  return image

colors = [[55, 255 , 28],
          [255, 123 , 253],
          [47, 248 , 255],
          [255, 181 , 29]]

image_path = '/content/drive/MyDrive/HK5/XLA/HairSegmentation/dataset/test/1.jpg'
image = cv2.imread(image_path)
n = int(image.shape[0]/224)
image = cv2.resize(image, (int(image.shape[1]/n), int(image.shape[0]/n)))
im = cv2.resize(image, (224, 224))
input_image = np.reshape(np.array(im, dtype=np.float32), [1, 224, 224, 3])/255
input_image = tf.convert_to_tensor(input_image, dtype=np.float32)

out = infer(input_image)
out = np.array(out['conv2d_46'])

mask = np.reshape(np.array(out, dtype=np.float32), [224, 224])
mask_ = mask *255.0
mask_ = cv2.resize(mask_, (image.shape[1], image.shape[0]))
mask_ = mask_.astype(np.uint8)
show = image.copy()
for color in colors:
  change = recoloring(image, mask_, color)
  show = np.hstack([show, change])
cv2_imshow(show)