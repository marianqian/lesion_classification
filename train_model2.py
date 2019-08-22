# USAGE: python train_model.py

# adapted from A. Rosebrock's ResNet malaria classification code
# to work with FFDM benchtop and MC-GPU images at FDA/OSEL/DIDSR

import matplotlib  # set matplotlib backend so figures can be saved in background
matplotlib.use('Agg')

#-----  import necessary packages

from   keras.preprocessing.image import ImageDataGenerator
from   keras.callbacks import LearningRateScheduler
from   keras.callbacks import ModelCheckpoint
from   keras.optimizers import SGD
from   resnet import ResNet
from   imutils import paths
from   utils import *

import numpy as np
import argparse
import os

#from   sklearn.metrics import classification_report
#from   pyimagesearch.resnet import ResNet
#from   pyimagesearch import config
#import matplotlib.pyplot as plt

#----- suppress TensorFlow warnings

os.environ['TF_CPP_MIN_LOG_LEVEL']= '3'

#----- construct argument parser and parse arguments

ap= argparse.ArgumentParser()
ap.add_argument('-p', '--plot', type=str, default='acc.png', help='path to output loss/accuracy plot')
args= vars(ap.parse_args())

#----- define total number of epochs to train for along with initial learning rate and batch size

NUM_EPOCHS= 40    # 120
INIT_LR=    0.01  # 0.1  0.05  0.01
BS=         32

def poly_decay(epoch):
        
	#----- initialize maximum number of epochs, base learning rate, and power of polynomial
        
	maxEpochs= NUM_EPOCHS
	baseLR=    INIT_LR
	power=     1.0
        
	#----- new learning rate based on polynomial decay
        
	alpha= baseLR*(1-(epoch/float(maxEpochs)))**power                
	return alpha

# def mean_subtract(img):
        
#         img= np.array(img)
#         np.subtract(img, 127.5)
#         return img / 255.0

#----- A. Rosebrack malaria dataset example

# img_siz= 64

# n_train= len(list(paths.list_images(config.TRAIN_PATH)))
# n_valid= len(list(paths.list_images(config.VALID_PATH)))
# #totalTest= len(list(paths.list_images(config.TEST_PATH)))

# trainAug= ImageDataGenerator(rescale= 1/255.0, rotation_range=20, zoom_range=0.05, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05, horizontal_flip=True, fill_mode='nearest')
# validAug= ImageDataGenerator(rescale= 1/255.0)

# trainGen= trainAug.flow_from_directory(config.TRAIN_PATH, class_mode='categorical', target_size=(img_siz, img_siz), color_mode='rgb', shuffle= True, batch_size=BS)
# validGen= validAug.flow_from_directory(config.VALID_PATH, class_mode='categorical', target_size=(img_siz, img_siz), color_mode='rgb', shuffle=False, batch_size=BS)
# #testGen= validAug.flow_from_directory(config.TEST_PATH, class_mode='categorical', target_size=(img_siz, img_siz), color_mode='rgb', shuffle=False, batch_size=BS)

#----- MC-GPU dataset

img_siz=     130
train_path= 'mass_vs_cyst/train'
valid_path= 'mass_vs_cyst/valid'

# train_datagen= ImageDataGenerator(preprocessing_function= mean_subtract)
# valid_datagen= ImageDataGenerator(preprocessing_function= mean_subtract)

train_datagen= ImageDataGenerator(rescale= 1/255.0)  #, vertical_flip=True)  # , horizontal_flip=True)
valid_datagen= ImageDataGenerator(rescale= 1/255.0)

trainGen= train_datagen.flow_from_directory(train_path, batch_size=BS, shuffle= True, color_mode='rgb', target_size=(img_siz, img_siz), class_mode='categorical')
validGen= valid_datagen.flow_from_directory(valid_path, batch_size=BS, shuffle=False, color_mode='rgb', target_size=(img_siz, img_siz), class_mode='categorical')

n_train= len(list(paths.list_images(train_path)))
n_valid= len(list(paths.list_images(valid_path)))

#----- initialize our ResNet model, compile it, and print its summary

model= ResNet.build(img_siz, img_siz, 3, 2, (3, 4, 6), (64, 128, 256, 512), reg= 0.0005)  # first three: width, height, and
print model.summary()                                                                     # depth= 3 (rgb) or 1 (grayscale)

opt= SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

#----- define our set of callbacks and fit model

chkpnt= ModelCheckpoint(filepath='resnet_mass_vs_cyst.h5', monitor= 'val_acc', verbose= 1, save_best_only= True)
callbacks= [LearningRateScheduler(poly_decay), chkpnt]

H= model.fit_generator(trainGen, steps_per_epoch= n_train // BS,
                       validation_data= validGen, validation_steps= n_valid // BS,
                       epochs= NUM_EPOCHS, callbacks= callbacks)

#----- reset testing generator and then use our trained model to
#----- make predictions on data

#print('[INFO] evaluating network...')
#testGen.reset()
#predIdxs= model.predict_generator(testGen, steps=(totalTest // BS) + 1)

#----- for each image in testing set we need to find index of
#----- label with corresponding largest predicted probability

#predIdxs= np.argmax(predIdxs, axis=1)

#----- show a formatted classification report

#print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))

#----- plot training loss and accuracy

#plot_performance(H)

N= NUM_EPOCHS
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(args['plot'])
