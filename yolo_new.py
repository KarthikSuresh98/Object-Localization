import pandas as pd
import numpy as np
from keras.layers import Conv2D , MaxPooling2D , Dense , Dropout , Flatten , LeakyReLU , Lambda, Input , BatchNormalization , GlobalAveragePooling2D
from keras.layers.merge import concatenate
from keras.models import Sequential , Model
import keras
import tensorflow as tf
from PIL import Image
from keras.models import model_from_json
import os
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator  
import matplotlib.pyplot as plt 

df = pd.read_csv(r'data/training_set.csv')
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
X = np.reshape(X, (24000, 480, 640, 1))
del images

df = pd.read_csv(r'data/training_set.csv', nrows=24000)
df['x1'] = df['x1']/640  
df['x2'] = df['x2']/640
df['y1'] = df['y1']/480
df['y2'] = df['y2']/480

Y = np.asarray(df['x1'])
Y = np.reshape(Y , (Y.shape[0] , 1))
Y = np.concatenate((Y , np.reshape(np.asarray(df['x2']) , (len(df['x2']) , 1))) , axis = 1)
Y = np.concatenate((Y , np.reshape(np.asarray(df['y1']) , (len(df['y1']) , 1))) , axis = 1)
Y = np.concatenate((Y , np.reshape(np.asarray(df['y2']) , (len(df['y2']) , 1))) , axis = 1)

X,Y = shuffle(X, Y, random_state =0)

def space_to_depth_x2(x):
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)

inpu = Input(shape = (480, 640, 1))
# Layer 1
x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(inpu)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 2
x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
x = BatchNormalization(name='norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 3
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
x = BatchNormalization(name='norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 4
x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
x = BatchNormalization(name='norm_4')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 5
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
x = BatchNormalization(name='norm_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 7
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
x = BatchNormalization(name='norm_7')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 8
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
x = BatchNormalization(name='norm_8')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 9
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
x = BatchNormalization(name='norm_9')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 10
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
x = BatchNormalization(name='norm_13')(x)
x = LeakyReLU(alpha=0.1)(x)

skip_connection = x

x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 11
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
x = BatchNormalization(name='norm_14')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 12
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
x = BatchNormalization(name='norm_20')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 13
skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
skip_connection = BatchNormalization(name='norm_21')(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
skip_connection = Lambda(space_to_depth_x2)(skip_connection)

x = concatenate([skip_connection, x])

# Layer 14
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
x = BatchNormalization(name='norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 15
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_23', use_bias=False)(x)
x = BatchNormalization(name='norm_23')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 16
x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_24', use_bias=False)(x)
x = BatchNormalization(name='norm_24')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 17
x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_25', use_bias=False)(x)
x = BatchNormalization(name='norm_25')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 18
x = Conv2D(4, (3,3), strides=(1,1), padding='same', name='conv_26', use_bias=False)(x)
x = BatchNormalization(name='norm_26')(x)
x = LeakyReLU(alpha=0.1)(x)

x = GlobalAveragePooling2D(data_format='channels_last')(x)
model = Model(inputs = inpu, outputs = x)

model.compile(loss = 'mean_squared_error' , optimizer = 'rmsprop' , metrics = ['accuracy'])
model.fit(X , Y , batch_size = 14 , epochs = 150 , validation_split = 0.1)

model_json = model.to_json()
with open("model_yolo_new_1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_yolo_new_1.h5")
print("Saved model to disk")
