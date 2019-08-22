#testing for mass and cyst differentiation with resnet
#!/usr/bin/python
#NOTEEEEEEEEEE: CHANGE PLOI*.PNG, #_HISTORY (PICKLE), AND #_TRY.H5 (WEIGHTS)
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense,Conv2D, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import matplotlib.pyplot as plt
import pickle
from keras.optimizers import SGD, Adam
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

#----- names to change every RUNS
weightnum = 'try5.h5'
plotnum = 'plot5.png'
picklenum = 'history5'

img_shape = 150
bs=32
seednumber = 42
reg=0.0005
bnEps=2e-5
bnMom=0.9
chanDim = -1
epochs = 100
init_lr = 0.01

#------ define total number of epochs to train for along with initial learning rate and batch size

def poly_decay(epoch):
    maxEpochs = epochs
    baseLR = init_lr
    power = 1.0
    #---- new learning rate based on polynomial decay
    alpha = baseLR * (1-(epoch/float(maxEpochs)))**power
    return alpha

#-------optional input_tensor, "to be use as image input for the model"
base_model = ResNet50(include_top = False, weights='imagenet', input_shape = (img_shape, img_shape, 3) )
#-------could not specify number of classes since include_top is false and weights is equal to image net
#-------decided not to put pooling layers as we would add another layer specifiying two classes
#-------include_top would not include the final pooling and fully connected layer in original model

x = base_model.output
x = Conv2D(32, (1, 1), use_bias=False, activation = 'relu', kernel_regularizer=l2(reg))(x)
x = Conv2D(128, (3,3), use_bias=False, activation = 'relu', kernel_regularizer=l2(reg))(x)
x = BatchNormalization(axis=-1, epsilon=bnEps, momentum=bnMom)(x)
x = GlobalAveragePooling2D()(x)
#x = Flatten()(x)
#------- added some other layers ending with sigmoid activation function --------
predictions = Dense(1, activation = 'sigmoid')(x) 
model = Model(inputs = base_model.input, outputs = predictions)

opt = Adam(lr = 1e-3, decay=1e-6)
#------ optimizer can be rmsprop or sgd 
model.compile(loss='binary_crossentropy', optimizer = opt, metrics =['accuracy'])

#------- finish creating model ---------
print('finish creating model')

#img_path = '/home/marian/ml/elephant.jpg'
#img = image.load_img(img_path, target_size=(150, 150))
#x = image.img_to_array(img)
#x=np.expand_dims(x, axis=0)
#x= preprocess_input(x)

#------- decided to use imagedatagenerator from keras in order to import from directory -------
#------- assuming that list in order is not actually the way it is displayed on the screen -------
#------- after reading in images, will split by validation and we will use that as the final accuracy score
#------- not doing backprop on validation, would test set be neccesary? will need to allocate other set of 
#        images before creating imagedatagenerator - will think about that beforehand 
#        
#train_path =r"/raidb/makeev/marks/lesions/valid/"
#valid_path = r"/raidb/makeev/marks/lesions/valid/"
train_path = r"/home/marian/ml/training/lesions/train/"
valid_path = r"/home/marian/ml/training/lesions/valid/"
traingen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True, vertical_flip=True)
validgen = ImageDataGenerator(rescale=1./255)

train  = traingen.flow_from_directory(train_path, target_size = (img_shape, img_shape), color_mode = 'rgb', 
                                     classes = ['cyst', 'mass'], class_mode = 'binary', batch_size = bs,
                                     shuffle = True, seed = seednumber) #, subset = 'training')

valid = validgen.flow_from_directory(valid_path, target_size = (img_shape, img_shape), color_mode = 'rgb',
                                    classes = ['cyst', 'mass'], class_mode = 'binary', batch_size = bs,
                                    shuffle = True, seed = seednumber) #, subset = 'validation')
#----- should i put valid shuffle false?? 
print('done reading data')
#X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.1, random_state = 42,
#                                                    shuffle = True)
#------validation would have been shuffled due to shuffle previously for train test split

callbacks = [LearningRateScheduler(poly_decay)]

#_------ have to edit model fit generator arguements - find lengths of the validation and train sets after flow from directory
H = model.fit_generator(train, steps_per_epoch = train.n // bs, validation_data = valid,
                       validation_steps = valid.n // bs, epochs = epochs, callbacks = callbacks) 
print('done training')

model.save_weights(weightnum)
print('done saving')
#-----steps per epoch declares when the epoch is finished 
#-----predict_generator 

#model.predict_generator(
# preds = model.evaluate(X_test, Y_test, batch_size = 32)
# print('Loss = ' + str(preds[0]))
# print('Test Accuracy = ' + str(preds[1]))

# #print('Predicted:', decode_predictions(preds, top=3)[0])

#------ plotting data

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
plt.savefig(plotnum)
print('done with plot')
#------ dumping records into dictionary
with open('/home/marian/ml/training/' + picklenum, 'wb') as file_pi:
    pickle.dump(H.history, file_pi)
print('done with pickle')
print('done with this run')

#----- evaluating from valid generator
print('evaluating')
evaluate = model.evaluate_generator(generator=valid, steps = valid.n // bs)
print('Accuracy: {}').format(evaluate[1])
print('Loss: {}').format(evaluate[0])


#----- in order to read pickle : pickle_file = open("second_history") metrics_dict = pickle.load(pickle_file)
#---- credit to pyimagesearch - how to use keras fit and fit generator a hands on tutorial

#TOTAL LOG OF TRIAL RUNS
#1. try1.h5 plot1.png - valid acc 0.8571 - highest but variable in a way between 0.500 after 100 epochs still 
#2. try2.h5 plot2.png history2 (changed dense to 1 node output and class type to binary) - more stable, ending with 0.65 to 0.7 
#   accuracy, highest valid acc 0.8571428656578064 (saved history in second_history pickle file) - training acc
#   really good so far, like 0,99, maybe since right now the valid set is full of 4 mm type images, hard to differentiate
#   *note now added the pickle history so we can see
#   - things to do: will be better with actual 10,000 iamges - should try randomization? idk or try to find if u can set the lr
#     right now just using rmsprop etc :) but other than that yea should be ok remmeber to put the graphs 
#     maybe can try diff layers and another btach or another dense layer 
#     can try with optimzing wd or regularization, batch size - decreasing momentum?? or increasing lr etc or use
#     adam instead - learningratescheduler
#3. try3.j5 plot3.png history3 (trying with rmsprop) - seems like doing sgd correctly - training loss is trying its best and acc is in 
#   high 90s, not sure why the validation loss is always around loss of 0.08 to 3.65 - ended with 0.5 - 0.78 - valid acc = 1.00 
#   [0.8977807760238647, 0.7142857313156128] loss and acc for evaluate_generator 
#4. try4.h5 plot4.png history4 (changed images to cars local and valid folders, not just made up of 4mm masses and added 
#   evaluate_generator function at the end) valid_acc = 1.0, very high loss... sometimes valid_acc even went to 0.000 lmfao
#5. try5... etc u get it (changed the imagedatagenerator, might add some data augmentation vertical hoirzontal flip and rescale
#   see how that does, also changed valid shuffle to true and implemented LearningRateScheduler, want to see if that helps at all and optimizer, testing with own imported from keras) 
#7. seventh try (see if gpu has enough memory to increase to 64 btachsize)
