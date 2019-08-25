#testing for mass and cyst differentiation with resnet
#!/usr/bin/python

from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense,Conv2D, Dropout, Flatten, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD, Adam
import numpy as np
import torch
from resnet import ResNet
from cyclical import *
from pyimagesearch.callbacks import TrainingMonitor
import matplotlib.pyplot as plt
import pickle
import os

number = '28'
print("[INFO] process ID: {}".format(os.getpid()))
#----- names to change every RUNS
weightnum = 'try'+number+'.h5'
plotnum = 'plot'+number
picklenum = 'history'+number
print('lets start number '+number)

img_size = 130
bs=64 #128 #32
seednumber = 42
reg=0.0005 
bnEps=2e-5
bnMom=0.9
chanDim = -1
epochs = 70 #50 #40 #50
init_lr = 0.01 #0.02 #0.058 #cant be large in the beginning because will be very difficult to raise accuracy (when loss is decreasing but 
                            #accuracy is staying still )
max_lr = 0.04 #0.588 #0.3467 #0.588 # 0.6 #found from find_lr, lowest and stable loss


sched = combine_scheds([0.3, 0.7], [sched_cos(init_lr, max_lr), sched_cos(max_lr, 0.001)])

def lr(epoch):
    epoch = float(epoch)/epochs
    return sched(epoch)

#---- from pyimagesearch
def poly_decay(epoch):
    maxEpochs = epochs
    baseLR = init_lr
    power = 1.0
    #---- new learning rate based on polynomial decay
    alpha = baseLR * (1-(epoch/float(maxEpochs)))**power
    return alpha
#----- from pyimagesearch
def step_decay(epoch):
    initAlpha = init_lr
    factor = 0.25
    dropEvery = 5
    alpha = initAlpha * (factor ** np.floor((1+epoch)/dropEvery))
    return float(alpha)

def plotdata():
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotnum+".png")
    plt.show()
    print('done with plot')
    
def savepickle():
    with open('/home/marian/ml/training/' + picklenum, 'wb') as file_pi:
        pickle.dump(H.history, file_pi)
    print('done with pickle')

def evaluate_model():
    print('evaluating')
    evaluate = model.evaluate_generator(generator=valid, steps = valid.n // bs)
    print('Accuracy: {}').format(evaluate[1])
    print('Loss: {}').format(evaluate[0])
    print(evaluate)

def readpickle(fname):
    pickle_file = open(fname) 
    metrics_dict = pickle.load(pickle_file)
    return metrics_dict

def readdata(trainf, validf):
    train_path = trainf
    valid_path = validf
    traingen = ImageDataGenerator(rescale=1./255)
    validgen = ImageDataGenerator(rescale=1./255)
    datagen = ImageDataGenerator()
    train  = traingen.flow_from_directory(train_path, target_size = (img_size, img_size), color_mode = 'rgb', 
                                     classes = ['cyst', 'mass'], class_mode = 'categorical', batch_size = bs,
                                     shuffle = True, seed = seednumber) 

    valid = validgen.flow_from_directory(valid_path, target_size = (img_size, img_size), color_mode = 'rgb',
                                    classes = ['cyst', 'mass'], class_mode = 'categorical', batch_size = bs,
                                    shuffle = True, seed = seednumber)
    print('done reading data')
    return train, valid

def createmodel(opt):
    model= ResNet.build(img_size, img_size, 3, 2, (3, 4, 6), (64, 128, 256, 512), reg= reg)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    #print(model.summary())
    print('finish creating model')
    return model

def createcallbacks():
    checkpoint = ModelCheckpoint("/home/marian/ml/training/best_"+weightnum, monitor="val_loss", save_best_only=True, verbose=1)
    figPath = os.path.sep.join(["monitor", plotnum+"_{}.png".format(os.getpid())])
    jsonPath = os.path.sep.join(["monitor", plotnum+"_{}.json".format(os.getpid())])
    callbacks = [ LearningRateScheduler(lr),checkpoint, TrainingMonitor(figPath, jsonPath=jsonPath)]
    return callbacks



train_path= r"/raidb/makeev/marks/mass_vs_cyst/all_lesions/train/"
valid_path= r"/raidb/makeev/marks/mass_vs_cyst/all_lesions/valid/"

#train, valid = readdata(train_path, valid_path)
#------- finish retrieving data --------

opt = SGD(lr=init_lr, momentum=0.9)
#model = createmodel(opt)
#------- finish creating model ---------

#**** have to call find_lr.py and find the max_lr and init_lr if want to use optimal lr 

callbacks = createcallbacks()
#------- finish creating callbacks -------

H = model.fit_generator(train, steps_per_epoch = train.n // bs, validation_data = valid,
                       validation_steps = valid.n // bs, epochs = epochs, callbacks = callbacks)
                       
print('done training')

model.save_weights(weightnum)
print('done saving')

#------ plotting data
plotdata()

#------ dumping records into dictionary
savepickle()

#----- evaluating from valid generator
evaluate_model()

print('done with this run ' + str(number))


