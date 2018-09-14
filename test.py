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
from keras.models import load_model
import keras.utils 
from common import *
import numpy as np 

def int_to_label(val):
	if val < 10:
		return str(val)
	else:
		return str(chr(ord('A') + val - 10))
	
#predicts label assuming all images in the directory are augmentation of the same test file	
def predict_label(model, dir):
	x, y = load_data(dir)
	y1 = model.predict(x)
	
	result = ""
	for place_prob in y1:
		stats = place_prob.sum(axis=0)
		result += int_to_label(np.argmax(stats))
		
	return result

#use the trained model to predict test captcha. 
model = load_model(os.path.join(model_dir, 'weights.hdf5'))
print(model.summary())

images = list_images(img_path_test)
count_succ = 0 

for img in images:
	prefix = img.split('.')[0]
	
	label = predict_label(model, os.path.join(img_path_test_augment, prefix))
	
	result = False 
	if [letter_to_label(ch) for ch in label] == [letter_to_label(ch) for ch in prefix]:
		result = True 
		count_succ += 1
		
	print("Guess {0}, while the tag {1}, which is {2}".format(label, prefix, result))
		
print("Guessed full captcha correctly {0} out of {1}, which is {2}%".format(count_succ, len(images), int(0.5 + count_succ * 100 / len(images))))