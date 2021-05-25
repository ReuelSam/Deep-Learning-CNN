#!/usr/bin/env python
# coding: utf-8

# In[87]:


# imports

import tensorflow as tf
from tensorflow.keras import datasets,layers, models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,Input, AveragePooling2D, Activation,Conv2D, MaxPooling2D, BatchNormalization,Concatenate
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


# In[88]:


class FashionMNIST_CNN():
    def __init__(self, epochs):
        self.class_names = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",                             "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
        self.epochs = epochs
        
    def fetch_data(self):
        dataset = tf.keras.datasets.fashion_mnist.load_data()
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = dataset
        self.print_shape()
        self.classes = np.unique(train_labels)
        self.nClasses = len(classes)
        print("Total number of clases: ", self.nClasses)
        print("Target Classes: ", self.classes)
        
    def print_shape(self):
        print("Training Data Shape: ", train_images.shape, train_labels.shape)
        print("Testing Data Shape: ", test_images.shape, test_labels.shape)
    
    def example_data(self, n):
        plt.figure(figsize=(10,10))
        for i in range(n):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[train_labels[i]]).set_color('white')
        plt.show()
        
    def model_start(self):
        self.preprocess()
        self.model_create()
        self.model_summary()
        var = self.model_compile()
        print("Model Creation Done")
    
    def preprocess(self):
        self.train_images = self.train_images.reshape((self.train_images.shape[0], 28, 28, 1))
        self.test_images = self.test_images.reshape((self.test_images.shape[0], 28, 28, 1))
        nRows, nCols, nDims = self.train_images.shape[1:]
        self.train_data = self.train_images.reshape(self.train_images.shape[0], nRows, nCols, nDims)
        self.test_data = self.test_images.reshape(self.test_images.shape[0], nRows, nCols, nDims)
        self.input_shape = (nRows, nCols, nDims)

        self.train_data = self.train_data.astype('float32')
        self.test_data = self.test_data.astype('float32')

        self.train_data /= 255
        self.test_data /= 255

        self.train_label_one_hot = to_categorical(self.train_labels)
        self.test_label_one_hot = to_categorical(self.test_labels)
        
    def model_create(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',                               padding='same', input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',                               padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D((2, 2)))
        
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.nClasses, activation='softmax'))
        
    def model_summary(self):
        self.model.summary()
            
    def model_compile(self):
        opt = optimizers.SGD(lr=0.01, momentum=0.9)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return 0
    
    def train(self):
        self.history = self.model.fit(self.train_data, self.train_label_one_hot, epochs=self.epochs,                                       validation_data=(self.test_data, self.test_label_one_hot))

    def model_evaluate(self):
        self.test_loss, self.test_acc = self.model.evaluate(self.test_data,  self.test_label_one_hot, verbose=2)

        print("Test Accuracy: ", self.test_acc)
        


# In[89]:


model = FashionMNIST_CNN(5)
model.fetch_data()


# In[90]:


model.example_data(15)


# In[91]:


model.model_start()


# In[92]:


model.train()


# In[93]:


model.model_evaluate()


# Add this to the class
# ```
#     def predict_new(self, image_path):
#         img=mpimg.imread(image_path)
#         ret_class = class_names[self.model.predict(np.expand_dims(img,axis=0)).argmax()]
#         print(ret_class)
#         return ret_class
#     
#     def model_save(self):
#         self.model.save('model')
#         
#     def model_load(self):
#         self.model = tf.keras.models.load_model('model')
# ```

# In[ ]:




