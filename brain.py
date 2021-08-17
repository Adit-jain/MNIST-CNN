import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout



class CNN:
    
    def __init__(self,training_set,validation_set):
        self.training_set = training_set
        self.validation_set = validation_set
        
    def create_model(self):
        self.network = tf.keras.models.Sequential()
        self.network.add(Conv2D(filters=100, kernel_size=3, activation='relu',input_shape=[28,28,1]))
        self.network.add(Conv2D(filters=100, kernel_size=3, activation='relu',input_shape=[28,28,1]))
        self.network.add(MaxPool2D((2,2),(2,2)))
        self.network.add(Flatten())
        self.network.add(Dense(units=600, activation='relu'))
        self.network.add(Dropout(0.5))
        self.network.add(Dense(units=10,activation='softmax'))
        print("Brain Created Successfully")
        
        
    def comp(self):
        self.network.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        print("Brain compilation successful")
        
    def train(self):
        self.network.fit(self.training_set,epochs=1000,validation_data=self.validation_set,use_multiprocessing=False)
        print("Brain training successful")
        
    def save(self):
        self.network.save('Saved model/')
        print("model saved")
        
    def load(self):
        self.network = tf.keras.models.load_model('Saved model/')
        print("Model Loaded")
        
        
        


