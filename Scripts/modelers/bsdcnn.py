import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, BatchNormalization, 
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Activation)
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Settings
kernel_size = (1,3)
scaler = MinMaxScaler()

# Knife
def shaper(Xs):
    X_train, X_valid, X_test = Xs
    print('in Xs:', X_train.shape, X_valid.shape, X_test.shape)
    
    # Scale (for 3D shaped X)            
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_valid = scaler.transform(X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    rows, cols = X_train.shape[1], X_train.shape[2]    
    input_shape = (rows, cols, 1) 

    X_train = X_train.reshape(X_train.shape + (1,)).astype('float32')
    X_valid = X_valid.reshape(X_valid.shape + (1,)).astype('float32')
    X_test = X_test.reshape(X_test.shape + (1,)).astype('float32')
    
    Xs_out = (X_train, X_valid, X_test)
    return Xs_out, input_shape
            
def bsdcnn(Xs, learning_rate, batch_size):  
    print('Initiated bsdcnn.py...')
    
    # plastic surgery
    Xs_out, input_shape = shaper(Xs)    
    
    # pick 1 model
    ### CHECKER
    #model = bsdcnn_signum(input_shape, batch_size)
    model = bsdcnn_relu(input_shape, batch_size)
    
    opt_adam = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=opt_adam, 
                  metrics=['binary_accuracy'],
                 )
    
    return Xs_out, model 

####################################################################################
## Custom Activation Function
# https://github.com/keras-team/keras/issues/7370

def sgn(x): # signum; piece wise function
    # https://en.wikipedia.org/wiki/Sign_function
    return tf.math.sign(x)


####################################################################################
## Builds
    
def bsdcnn_signum(input_shape, batch_size):
    # https://www.researchgate.net/publication/349293613_Binary_Single-Dimensional_Convolutional_Neural_Network_for_Seizure_Prediction
    
    get_custom_objects().update({'sgn': Activation(sgn)})
    
    # ver 1: 2D_sgn
    model = Sequential()
    
    model.add(Conv2D(16, (1,5), activation='relu',input_shape=input_shape))
    model.add(Conv2D(16, (1,5), strides=2, activation='relu',padding="valid"))
    model.add(BatchNormalization())  
    
    model.add(Conv2D(32, (1,5), padding="same", activation=sgn))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (1,5), strides=2, padding="same", activation=sgn))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, (1,10), padding="same", activation=sgn))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (1,10), strides=2, padding="same", activation=sgn))
    model.add(BatchNormalization())   
    
    model.add(Conv2D(128, (2,1), padding="same", activation=sgn))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (2,1), strides=2 ,padding="same", activation=sgn))
    model.add(BatchNormalization())  
    
    model.add(Conv2D(256, (2,1), padding="same", activation=sgn))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (2,1), strides=2, padding="same", activation=sgn))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def bsdcnn_relu(input_shape, batch_size):
    # https://www.researchgate.net/publication/349293613_Binary_Single-Dimensional_Convolutional_Neural_Network_for_Seizure_Prediction
    
    get_custom_objects().update({'sgn': Activation(sgn)})
    
    # ver 1: 2D_sgn
    model = Sequential()
    
    model.add(Conv2D(16, (1,5), activation='relu',input_shape=input_shape))
    model.add(Conv2D(16, (1,5), strides=2, activation='relu',padding="valid"))
    model.add(BatchNormalization())  
    
    model.add(Conv2D(32, (1,5), padding="same", activation=sgn))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (1,5), strides=2, padding="same", activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, (1,10), padding="same", activation=sgn))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (1,10), strides=2, padding="same", activation='relu'))
    model.add(BatchNormalization())   
    
    model.add(Conv2D(128, (2,1), padding="same", activation=sgn))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (2,1), strides=2 ,padding="same", activation='relu'))
    model.add(BatchNormalization())  
    
    model.add(Conv2D(256, (2,1), padding="same", activation=sgn))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (2,1), strides=2, padding="same", activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    
    return model