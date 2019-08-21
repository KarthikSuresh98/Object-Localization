import pandas as pd
import numpy as np
from keras.layers import Conv2D , MaxPooling2D , Dense , Dropout , Flatten , LeakyReLU , Lambda, Input , BatchNormalization , GlobalAveragePooling1D
from keras.layers.merge import concatenate
from keras.models import Sequential , Model
import keras
import tensorflow as tf
from PIL import Image
from keras.models import model_from_json
import os
from keras.preprocessing.image import ImageDataGenerator  
import matplotlib.pyplot as plt 

df = pd.read_csv(r'data/test.csv')
s = df['image_name']

images = []
for i in range(len(s)):
    loc = r'data/images/'+s[i]
    img1 = Image.open(loc)
    nx , ny = img1.size
    img = np.asarray(img1.resize((int(nx*1),int(ny*1)),Image.BICUBIC))
    images.append(img)
    if(i%500 == 0):
        print(i)
    img1.close()
X = np.asarray(images)
X = np.reshape(X, (24045, 480, 640,1))
del images


def space_to_depth_x2(x):
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)

json_file = open('model_yolo_new_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json , custom_objects={"tf": tf})
loaded_model.load_weights('model_yolo_new_1.h5')
print("Loaded model from disk")

pred = loaded_model.predict(X, batch_size = 32)
df['x1'] = pred[:, 0]*640
df['x2'] = pred[:, 1]*640
df['y1'] = pred[:, 2]*480
df['y2'] = pred[:, 3]*480
df.to_csv("submission.csv")
