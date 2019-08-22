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
import os 
from pyimagesearch.callbacks import TrainingMonitor
from keras import backend as K
import matplotlib.pyplot as plt
import pickle
#import cv2
#from sklearn.model_selection import train_test_split
number = '24'
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
epochs = 40 #50
init_lr = 0.01
max_lr = 0.6

#------ from fastai defining cyclical 

sched = combine_scheds([0.3, 0.7], [sched_cos(init_lr, max_lr), sched_cos(max_lr, 0.001)])

def e(epoch):
    epoch = float(epoch)/epochs
    return sched(epoch)

#------ define total number of epochs to train for along with initial learning rate and batch size
#---- from pyimagesearch
def poly_decay(epoch):
    maxEpochs = epochs
    baseLR = init_lr
    power = 1.0
    #---- new learning rate based on polynomial decay
    alpha = baseLR * (1-(epoch/float(maxEpochs)))**power
    return alpha

def step_decay(epoch):
    initAlpha = init_lr
    factor = 0.25
    dropEvery = 5
    alpha = initAlpha * (factor ** np.floor((1+epoch)/dropEvery))
    return float(alpha)

#-------optional input_tensor, "to be use as image input for the model"
#base_model = ResNet50(include_top = False, weights='imagenet', input_shape = (img_size, img_size, 3) )
#-------could not specify number of classes since include_top is false and weights is equal to image net
#-------decided not to put pooling layers as we would add another layer specifiying two classes
#-------include_top would not include the final pooling and fully connected layer in original model

#x = base_model.output
##x = Conv2D(2048, (1, 1), use_bias=False, activation = 'relu', kernel_regularizer=l2(reg))(x)
#x = Conv2D(2048, (3,3), use_bias=False, activation = 'relu', kernel_regularizer=l2(reg))(x) #l2 is weight decay (sum of squared weights)
#x = BatchNormalization(axis=-1, epsilon=bnEps, momentum=bnMom)(x)
##x = AveragePooling2D(padding = 'same')(x)
#x = GlobalAveragePooling2D()(x)
##x = Flatten()(x)
##------- added some other layers ending with sigmoid activation function --------
#predictions = Dense(2, activation = 'sigmoid')(x) 
#model = Model(inputs = base_model.input, outputs = predictions)

#opt = Adam(lr = init_lr)
opt = SGD(lr=init_lr, momentum=0.9)
#------ optimizer can be rmsprop or sgd 
#model.compile(loss='binary_crossentropy', optimizer = opt, metrics =['accuracy'])


#------ another model architecture using resnet
#model= ResNet.build(img_size, img_size, 3, 2, (3, 4, 6), (64, 128, 256, 512), reg= reg)
#model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#print(model.summary())

#------- finish creating model ---------
print('finish creating model')

#------- decided to use imagedatagenerator from keras in order to import from directory -------
#------- assuming that list in order is not actually the way it is displayed on the screen -------
#------- after reading in images, will split by validation and we will use that as the final accuracy score
#------- not doing backprop on validation, would test set be neccesary? will need to allocate other set of 
#        images before creating imagedatagenerator - will think about that beforehand 
#        
#train_path =r"/home/makeev/ml/dl-medical-imaging/mass_vs_cyst/train/"
#valid_path = r"/home/makeev/ml/dl-medical-imaging/mass_vs_cyst/valid/"
#train_path = r"/raidb/makeev/marks/marian_tests/lesions/train/"
#valid_path = r"/raidb/makeev/marks/marian_tests/lesions/valid/"
#train_path = r"images/train/"
#valid_path = r"images/valid/"
train_path= r"/raidb/makeev/marks/mass_vs_cyst/all_lesions/train/"
valid_path= r"/raidb/makeev/marks/mass_vs_cyst/all_lesions/valid/"
traingen = ImageDataGenerator(rescale=1./255)
validgen = ImageDataGenerator(rescale=1./255)
datagen = ImageDataGenerator()
#train  = traingen.flow_from_directory(train_path, target_size = (img_size, img_size), color_mode = 'rgb', 
                                     #classes = ['cyst', 'mass'], class_mode = 'categorical', batch_size = bs,
                                     #shuffle = True, seed = seednumber) #, subset = 'training')

#valid = validgen.flow_from_directory(valid_path, target_size = (img_size, img_size), color_mode = 'rgb',
                                    #classes = ['cyst', 'mass'], class_mode = 'categorical', batch_size = bs,
                                    #shuffle = True, seed = seednumber) #, subset = 'validation')
#----- should i put valid shuffle false?? 
print('done reading data')

#------validation would have been shuffled due to shuffle previously for train test split
checkpoint = ModelCheckpoint("/home/marian/ml/training/best_"+weightnum, monitor="val_loss", save_best_only=True, verbose=1)
figPath = os.path.sep.join(["monitor", plotnum+"_{}.png".format(os.getpid())])
jsonPath = os.path.sep.join(["monitor", plotnum+"_{}.json".format(os.getpid())])
callbacks = [ LearningRateScheduler(e),checkpoint, TrainingMonitor(figPath, jsonPath=jsonPath)]

#_------ have to edit model fit generator arguements - find lengths of the validation and train sets after flow from directory
H = model.fit_generator(train, steps_per_epoch = train.n // bs, validation_data = valid,
                       validation_steps = valid.n // bs, epochs = epochs, callbacks = callbacks)
                       valid.n // bs, epochs = epochs, callbacks = callbacks) 
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
plt.savefig(plotnum+".png")
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
print(evaluate)
print(number)


#----- in order to read pickle : pickle_file = open("second_history") metrics_dict = pickle.load(pickle_file)
#---- credit to pyimagesearch - how to use keras fit and fit generator a hands on tutorial
"""
TOTAL LOG OF TRIAL RUNS
1. try1.h5 plot1.png - valid acc 0.8571 - highest but variable in a way between 0.500 after 100 epochs still 
2. try2.h5 plot2.png history2 (changed dense to 1 node output and class type to binary) - more stable, ending with 0.65 to 0.7 
  accuracy, highest valid acc 0.8571428656578064 (saved history in second_history pickle file) - training acc
  really good so far, like 0,99, maybe since right now the valid set is full of 4 mm type images, hard to differentiate
  *note now added the pickle history so we can see
  - things to do: will be better with actual 10,000 iamges - should try randomization? idk or try to find if u can set the lr
    right now just using rmsprop etc :) but other than that yea should be ok remmeber to put the graphs 
    maybe can try diff layers and another btach or another dense layer 
    can try with optimzing wd or regularization, batch size - decreasing momentum?? or increasing lr etc or use
    adam instead - learningratescheduler
3. try3.h5 plot3.png history3 (trying with rmsprop) - seems like doing sgd correctly - training loss is trying its best and acc is in 
  high 90s, not sure why the validation loss is always around loss of 0.08 to 3.65 - ended with 0.5 - 0.78 - valid acc = 1.00 
  [0.8977807760238647, 0.7142857313156128] loss and acc for evaluate_generator 
4. try4.h5 plot4.png history4 (changed images to cars local and valid folders, not just made up of 4mm masses and added 
  evaluate_generator function at the end) valid_acc = 1.0, very high loss... sometimes valid_acc even went to 0.000 lmfao
5. try5... etc u get it (changed the imagedatagenerator, might add some data augmentation vertical hoirzontal flip and rescale
  see how that does, also changed valid shuffle to true, implemented LearningRateScheduler, want to see if that helps at all and optimizer
  testing with own imported from keras adam) (restarted with just using the data augmentation without lr change since 
  the model was not training well, maybe lr was too small? rmsprop as optimizer) -(RESULTS - HMM THIS IS VERY WEIRD... 
  after trying the augmentation just didnt end up converging, stuck at loss approx 0.7 ish 0.5 accuracy and after removing horizontal/vertical flip
  and the rotation - only left rescale - it converged pretty quickly... why is that doing worse with augmentation wtf. ) valid acc 0.89 around,
  but still really bouncy but didnt go to 0 (evaluate_generator  [0.968887448310852, 0.8125] loss and acc) 
5. fifth try (finally implemented learning rate scheduler, changed from rmsprop to adam with init lr 0.01 lets see how it does) 
  hm,, loss is really slow - took approx 70 epochs until reach 90 compared to like 20... could have larger lr in the beginning? 
  valid_loss showed a bit overfitting with a bunch of epochs with rising losses and then came back down again during the last few - 
  valid acc also 0.5-0.89, valid loss 0.2-0.4, training up to 1.000 and 0.04 [0.4183688759803772, 0.875] evaluate loss and acc !! 
6. sixth try (see if gpu has enough memory to increase to 64 btachsize) meh not much better - evaluation [1.4644379615783691, 0.6274510025978088] 
  able to converge with the training data, not much for the validation, might want to keep at 32 until more data
7. seventh try (if increase init_lr to 0.1 instead of 0.01, bs - 32) [1.3681305646896362, 0.7368420958518982] for evaluate, not sure if it looks 
  much better :( didnt go into the 0.0s until 99 epoch could because the large lr was too big to converge and need a smaller one in the beginning
  better to stick with 0.01 for now 
8. eigth (didnt change much shorter epoch 10 but wanted to test out trainingmonitor, modelcheckpoint - names with monitor_ and best_)
  now plotnum_pid.json and png holds the image and graph, same as plotnum.png but now you can check up on it - best_weights.h5 - able to store best 
  loss weights and created pyimagesearch modules and libraries yay
9. ninth (hopefully able to train with all 10,000 images - split validation into 470 so about 0.05 percent maybe more not sure, going to try without 
  lr scheduler and regular rmsprop or sgd with momentum 0.01 lr) - RAN WITH ABOUT ALL OF CYST AND HALF OF MASS IMAGES using SGD OPT init 
  lr = 0.01 and mom = 0.9 with lr LearningRateScheduler
10. without lr sched same as before - ummmmm validation loss IS HORRIBLE but the training can converge really well and fast, validation loss increases 
  so much i dont understand why
11. trying to see if larger bs up to 64 would generalize better..... and with poly decay and halved the epochs, set valid shuffle to false 
  (possibility of overfitting??) yep nope def overfitting ending evaluate generator ended up with 0.5 accuracy
12. increase bs again to 128. increase init lr to 0.1 with sgd opt (256 ranout memory) trying with all 9400 images 
  [8.846018736822563, 0.4415204678362573] evaluate generator overfitting again )using sgd and learning rate sched 
13. using dr makeev arcitecture less complex resnet using sgd and learning rate sched/same parameters epoch 50 init_lr= .01
  changed model to end with 2048 plus another conv2d with 2048 end with dense layer 3 by 3 kernel size one layer (before was 32 (1,1) and 128 (3.3)
  couldnt finish training not doing well definitely overfittin again (didnt save history or end weights whoops)
14. using same model as dr makeev did from the resnet module from pyimagesearch - changed generators to categorical instead of binary 
  (bs = 64, decreased from 128) Accuracy: 0.504926109255 Loss: 1.75148225535 - loss for training around 0.5 so wasnt specifically overfitting in any way....
  but valid loss exploded to 6.5 
15. 130 size - worked alot better on new images?? not sure why ... should be the exact same just that image size is larger - [0.7246856008257184, 0.9434523809523809] 
  yay! valid acc very good smooth stable OHHHH THE VALID IS ONLY OF 7 AND 8 MM MASSES SO VERY EASY.......or is it actually that reason?
16. back to 150 size just in case - shouldnt be that big of a difference... exploded again to 7.0 :( [6.529020180255909, 0.5689655172413793]

DISCUSSION AND THOUGHTS: i dont think the model overfitted with 130 size since it was able to get validation images correct but 
  i dont know why when you add the 4 mm size validation images it does not work anymore and goes back to guessing... 
  both are able to train but is it because with 150 images NOW It tends to overfit?? but there should be absolutely no reason why...

17. trying with 64 as bs, changed momentum in opt to 0.95 and removed last 512, 6filter stage (size 150) (sigh didnt want to make it less generalized... 
  idk how else to do tho training acc went to 1.0000 around last few epochs too so its not like it cant differentiate
   loss: 0.6391 - acc: 0.8076 - val_loss: 1.7796 - val_acc: 0.5394 (after this started to blow up to 6.7 )
18. (makeev architecture) copied 130 images to my folder and added 4mm images to validatin set, see how that does if this doesnt work- added back 512 filters, 
  made mom = 0.9 not doing horribly?? might want to figure out why the images i generated are not successfully working...  
  [1.0031380487042805, 0.8465116254118986] much better, lower 10% from try 15  due to inclusion of the 4 mm validation data - training acc = 1.00000 around epoch 27
19. trying to use learning rate bs = 64, changed init_lr = 0.2 (oops was kept constant)
  https://github.com/keras-team/keras/issues/3755 (!!) [0.625635800250741, 0.8255813945171445] lower loss in general?? 
  but weird large losses up to 1.5 like in epochs in between... and accuracy overall not that great
  might have started to overfit a little looking at the graph, but would want it to 1. generalize better to get that accuracy up and to 2. prevent more overfitting
20. NOW USING CYCLICAL LEARNING RATE (combine_scheds([0.3, 0.7], [sched_cos(0.2, 0.6), sched_cos(0.6, 0.001)])) init lr = 0.2, bs = 64 
  (forgot to change to float so constant at 0.2) [0.45377633821132574, 0.8395348812258521] oi hm pretty good! but started at really high loss not sure why (8.2) 
  other trainings not as bad.. not sure why since try 19 used 0.2 as init lr and did not have 8 as initial loss...... 
  lower loss, slightly better acc than before, want to see how better it goes - doesnt seem like overfitting for now since valid loss has been constantly
  improving until the last epoch (more specific/complex structure?) ******* 
20b going to further train for 10 more epochs at cyclical but with smaller lr combine_scheds([0.3, 0.7], [sched_cos(0.001, 0.01), sched_cos(0.01, 0.001)])  
  = [0.4933106585990551, 0.8325581409210383] not that much improvement (didnt save values in pickle file) - will continue with 20 more epochs with little 
  larger max_lr = 0.05 - better accuracy, high loss?? (https://forums.fast.ai/t/validation-loss-vs-accuracy/8514/4) 
  [0.4918557131013205, 0.8697674440783124] model better training with set threshold that we are giving it, but further away from the set truth? higher lr allowed
  model to have higher training loss for a bit, then converged but not sure if it exactly overfit... 
    20c finding learning rate using LR finder saved in history20clr and onecycle hmmm so suppsoed to do this before training and then trian same model using new lr
        one cycle algorithm that you got - looking at clr.png, looks like -2.35 was most 0.00446683592151 was most steep, so using that for next 15 ep training for 
        max lr- prob doing in wrong order but like its what we do to see if it works out :) 
    20d. testing out with 0.00446683592151 lr 15 epochs: new combine_scheds([0.3, 0.7], [sched_cos(0.0004, 0.00446683592151), sched_cos(0.00446683592151, 0.00004)])
        with own history20d, using monitor with 20b (graph), checkpoints with 20b (bestloss), loaded in the best_try20b.h5 weights since the finding lr kinda messed 
        up the model (not supposed to change the weights after doing lr find but we did the order wrong) [0.40088475194088247, 0.8674418585245] highest evaluate
        not sure if model can get any higher due to the limit - might need to start over with actually following procedure
        ** all the experiments didnt neccesarily improve that much only improved val loss by a bit, training convergence kinda slow since we had a small lr ~ ended 
        with [0.16962024244591273, 0.9703686948654151] training acc, might want to get it to 1,000 but that might over fit :( 
21. using new all_lesions data from raidb just to see how new validation set is actually wholly representative of the dataset we have now (or was working on for long 
    time. using same archetiturea and everything as try 20 EXCEPT making init_lr as 0.01 (original was 0.2) in order to prevent the shoot up to 8.2 training loss 
    in the beginning [0.21030441778046743, 0.9553571428571429] oh ho lowest val loss was 0.18861 during 38 epoch, so no overfitting. most likely due to the lr 
22. will run with the images originally used (images/train or valid on cars) [0.32047259211540224, 0.9255813975666844] - maybe a little less since there were more 
    images of 4mm and harder?? the other dataset is probably more realistic and generalizes/shows more representative dataset so will USE THAT FROM NOW ON 
23. using the raidb all_lesions just to see how well it reproduces once and twice again .. [0.656611008303506, 0.7648809523809523] ummm ok running again :,(
24. finding lr finder - one epoch to find the lr and changing the sched etc. STILL NEED TO DO 



21. try to add more filters per conv layer (change last 256, 512 to 512, 1024) since that shouldnt cause more chaotic landscapes? 
22. try to use the lr finder as well as the one cycle stuff etc 
22. use rmsprop or Adam? optimizers or change structure a bit weight decay?  https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/ (change conv to 0.0005 or dense to 1e-4 to 1e-6) or implement cyclical momentum - ORRRRRR change learning rate larger lr in the beginning / find optimal momentum and weight decay along with cyclical lr (find optimal max lr) 

WHAT TO DO WHEN U GET BACK ON MONDAY: THE IMAGES FOR MASS will be split bewen 8 mm and 7mm not made images yet. you can remake all of them by removing them from their 
individual folders and placing them in the BIG mass50 or mass26 folder OK and then create the images like u did with the cysts, having 4700 images for 50 and 26 kvp
INSIDE THEIR RESPECTIVE FOLDERS after creating the two SEPARATELY put them together USING THE 3 CHANNEL PYTHON PROGRAM AND SHOULD GET 4700 50+26 IMAGES THANK YOU 

do you want to figure out why the 150 size images do not generalize well or if it has something to do with the images i generated - if i did something wrong 
"""
