from os import listdir
import keras
from keras.preprocessing import image
import os.path
import numpy 
from keras.datasets import mnist
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from common import *

train, labels_keras = load_data(img_path_train_augment)

input_shape = (num_rows, num_cols, 3)
inputs = Input(shape=input_shape)

#Few ideas were taken into consideration: 
#In this problem, the recognition letter task is combined with placement recognition. We want to share the recognition task among all layer, 
#and only give weight to some probable placement only in the last possible moment. This is why only one last layer should is fully connected layer. 
#Another point is the amount of data we have is limited this is why I used another "compressing" filter to reduce amount of parameters before the last layer (idea from sqeezenet). 
#The other considerations were VGG reduction (increase the amount of layers after max pooling), and amount of parameters which should be of the same order of magnitude as the data (no big networks). 
#This networked looked good in general giving ~90-95% to all digits except the first (not sure why the first is so not stable). 
x = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(48, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = Dropout(0.5)(x)
x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x1 = Dense(num_classes, activation='softmax')(x)
x2 = Dense(num_classes, activation='softmax')(x)
x3 = Dense(num_classes, activation='softmax')(x)
x4 = Dense(num_classes, activation='softmax')(x)
x5 = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=[x1, x2, x3, x4, x5])
print(model.summary())

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

make_sure_dir(model_dir)
checkpointer = ModelCheckpoint(filepath=os.path.join(model_dir, 'weights.hdf5'), verbose=1, save_best_only=True)

#used GPU to train
model.fit(train, [labels_keras[0], labels_keras[1], labels_keras[2], labels_keras[3], labels_keras[4]],
          batch_size=8,
          epochs=100,
          verbose=1,validation_split=0.05,callbacks=[checkpointer])