from os import listdir
import keras
from keras.preprocessing import image
import os.path
import numpy 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import os 
from common import *

list_images = list_images(img_path)
print("validating images...")
invalid = []

for f in list_images:
	for i in range(5):
		label = letter_to_label(f[i].lower())
		
		if label < 0 or label > 15:
			print("the file {0} is invalid - please fix before continue".format(f))
			invalid.append(f)
			break
			
if len(invalid) > 0:

	to_delete = input("Enter 'y' if you want to delete invalid files")
	
	if to_delete:
		for f in invalid:
			os.remove(os.path.join(img_path, f))
	else:
		print("exiting the application - failed data was detected, please fix and try again")
		exit(1)

make_sure_dir(img_path_train)
make_sure_dir(img_path_train_augment)
make_sure_dir(img_path_test)
make_sure_dir(img_path_test_augment)

print("Splitting test and train...")
split_test_train(200)
print("Augmenting train...")
augment_train_dir(img_path_train, img_path_train_augment, 50)
print("Augmenting test...")
augment_test_dir(img_path_test, img_path_test_augment, 10)

