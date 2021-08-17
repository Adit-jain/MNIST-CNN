import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as k
import matplotlib.pyplot as plt
#from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from keras.utils import np_utils
from brain import CNN

dataset = pd.read_csv("train.csv")
X_train = dataset.iloc[:,1:].values
Y_train = dataset.iloc[:,0].values




print("Number of training examples : ",X_train.shape[0])
print("Number of classes : ",len(np.unique(Y_train)))



Y_train = np_utils.to_categorical(Y_train).astype('int32')



X_train = X_train.reshape(42000,28,28,1)





train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.20,
    height_shift_range=0.20,
    validation_split=0.1,
    rescale=1./255)


train_datagen.fit(X_train)


training_set = train_datagen.flow(X_train,Y_train,batch_size=32,subset='training')
print("Training data created")
validation_set = train_datagen.flow(X_train,Y_train,batch_size=32,subset='validation')
print("Validation data created")



test_dataset = pd.read_csv("test.csv")
X_test = test_dataset.values
X_test = X_test.reshape(-1,28,28,1).astype('float32')

test_datagen = ImageDataGenerator(rescale=1./255)

test_datagen.fit(X_test)

test_set = test_datagen.flow(X_test,batch_size=32)


print("Test data created")

# for x_batch in test_set:
#     for i in range(32):
#         plt.subplot(4,8,i+1)
#         plt.imshow(x_batch[i].reshape(28,28),cmap=plt.get_cmap('gray'))
#     plt.show()
#     break
    
print("preview done")

##Try with CNN

Brain = CNN(training_set,validation_set)
Brain.create_model()

Brain.comp()

Brain.train()

#Brain.save()
#print("Brain stored")

##To load a stored mode
#Brain.load()

Brain.network.summary()

Y_pred = Brain.network.predict(X_test)




