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
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random 
from shutil import copyfile


num_rows = 30
num_cols = 66
num_classes = 16

img_path               = 'data/labeled_captchas'
img_path_train         = 'data/train'
img_path_train_augment = 'data/train_aug'
img_path_test          = 'data/test'
img_path_test_augment  = 'data/test_aug'
model_dir              = 'models'

def letter_to_label(letter):
	
	#Very ugly hack. Taggers should label the data properly. 
	if letter == "o":
		return 0
	
	if letter == "l":
		return 1
	
	if letter == "g" or letter == "q":
		return 0
	
	if letter == "p":
		letter = "f"
	
	if letter == "s":
		return 5
	
		
	if ord(letter) <= ord('9') and ord(letter) >= ord('0'):
		return ord(letter) - ord('0')
	else:
		return 10 + ord(letter) - ord('a') 

def list_images(dir):
	return [f for f in os.listdir(dir) if f.endswith('.png')]

def split_test_train(num_test):
	list = list_images(img_path)
	random.shuffle(list)
	
	for i, fname in enumerate(list):
		if i < num_test:
			copyfile(os.path.join(img_path, fname), os.path.join(img_path_test,fname))
		else:
			copyfile(os.path.join(img_path, fname), os.path.join(img_path_train,fname))

#loads data with labels using file names 
def load_data(dir):
	images = list_images(dir)
	num_files = len(images)
	
	train = numpy.zeros((num_files, num_rows, num_cols, 3))
	labels = numpy.zeros((5, num_files))
	labels_keras = numpy.zeros((5, num_files, num_classes))

	for i, fname in enumerate(images):
		train[i] = image.load_img(os.path.join(dir, fname), target_size=(num_rows, num_cols))

		fname = fname.lower()
		
		for j in range(5):
			
			labels[j][i] = letter_to_label(fname[j])
			
			if labels[j][i] > 15 or labels[j][i] < 0:
				print("This name is invalid, please fix or erase this file", fname)
			
		if len(images) > 2000:
			if int(i * 100 / len(images)) != (int((i + 1) * 100 / len(images))):
				print("loaded {0}%".format((int((i + 1) * 100 / len(images)))))
				
	train = train.astype('float32')
	train /= 255
	
	for j in range(5):
		labels_keras[j] = keras.utils.to_categorical(labels[j], num_classes)

	return train, labels_keras

def augment_image(img_name, prefix, num_copies, target_dir, datagen):
	
	img = load_img(img_name, target_size = (75, 165)) # this is a Numpy array with shape (3, 75, 165)
	x = img_to_array(img)  
	x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 75, 165)

	
	# the .flow() command below generates batches of randomly transformed images
	# and saves the results to the `preview/` directory
	i = 1
	for batch in datagen.flow(x, batch_size=1, 
							  save_to_dir=target_dir, save_prefix=prefix, save_format='png'):
		i += 1
		if i > num_copies:
			break  # otherwise the generator would loop indefinitely

def make_sure_dir(dir_name):
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)
	

def get_datagen():
	return ImageDataGenerator(
        rotation_range=3,
        width_shift_range=0.07,
        height_shift_range=0.07,
        shear_range=0.05,
        zoom_range=[1, 1.2],
		channel_shift_range = 0.5,
        horizontal_flip=False,
        fill_mode='nearest')

def augment_train_dir(source_dir, target_dir, num_copies):
	
	list = list_images(source_dir)
	datagen = get_datagen()
	
	for fname in list:
		augment_image(os.path.join(source_dir, fname), fname.split('.')[0], num_copies, target_dir, datagen)

def augment_test_dir(source_dir, target_dir, num_copies):
	
	list = list_images(source_dir)
	datagen = get_datagen()
	
	for fname in list:
		f_target_dir = os.path.join(target_dir, fname.split('.')[0])
		make_sure_dir(f_target_dir)
		
		augment_image(os.path.join(source_dir, fname), fname.split('.')[0], num_copies, f_target_dir, datagen)
			