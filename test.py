import tensorflow as tf
import numpy as np
from keras import backend as K
from sklearn.metrics import jaccard_score as iou
from sklearn.metrics import accuracy_score

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

path_face = '/content/drive/MyDrive/HK5/XLA/HairSegmentation/dataset/test_face'
path_mask = '/content/drive/MyDrive/HK5/XLA/HairSegmentation/dataset/test_mask'
list = os.listdir(path_face)

acc = []
meaniou = []

x = 0
for i in list:
  print(x)
  x = x + 1
  image_path = path_face + '/' + i
  mask_path = path_mask + '/' +i
  image = cv2.imread(image_path)
  mask_true = cv2.imread(mask_path)
  mask_true = cv2.cvtColor(mask_true, cv2.COLOR_BGR2GRAY)
  _,mask_true = cv2.threshold(mask_true, 50, 255, cv2.THRESH_BINARY)
  im = cv2.resize(image, (224, 224))

  input_image = np.reshape(np.array(im, dtype=np.float32), [1, 224, 224, 3])/255

  input_image = tf.convert_to_tensor(input_image, dtype=np.float32)

  out = infer(input_image)
  out = np.array(out['conv2d_46'])

  img = image.copy()

  mask = np.reshape(np.array(out, dtype=np.float32), [224, 224])
  mask = mask *255.0
  mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
  mask = mask.astype(np.uint8)
  mask = cv2.resize(mask, (mask_true.shape[0], mask_true.shape[1]))
  _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

  a = accuracy_score(mask_true.flatten(), mask.flatten())
  acc.append(a)
  jac = iou(mask_true.flatten(), mask.flatten(), average=None)
  meaniou.append(np.mean(jac))

print(np.mean(acc))
print(np.mean(meaniou))