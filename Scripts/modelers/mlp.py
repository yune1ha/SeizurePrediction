''' Ref
* https://towardsdatascience.com/deep-learning-model-interpretation-using-shap-a21786e91d16
''' 

import tensorflow as tf
tf.compat.v1.disable_v2_behavior() 
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
scaler = MinMaxScaler()


# Knife
def shaper(Xs):
    X_train, X_valid, X_test = Xs
    print('in Xs:', X_train.shape, X_valid.shape, X_test.shape)
    

    #Scale (for 3D shaped X)            
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_valid = scaler.transform(X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    print('@@@ IN shape:', X_train.shape)

    num_pixels = X_train.shape[1] * X_train.shape[2]

    X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
    X_valid = X_valid.reshape((X_valid.shape[0], num_pixels)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

    print('@@@ OUT shape:', X_train.shape)
    
    
    Xs_out = (X_train, X_valid, X_test)
    return Xs_out, num_pixels
            
def mlp(Xs, learning_rate, batch_size):  
    print('Initiated mlp.py...')
    
    # plastic surgery
    Xs_out, num_pixels = shaper(Xs)    
    
    # pick 1 model
    print('Modeling...')
    model = b1_mlp(num_pixels, batch_size)
    
    opt_adam = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    print('Compling...')
    model.compile(loss='binary_crossentropy', 
                  optimizer=opt_adam, 
                  metrics=['binary_accuracy'],
                 )
    
    return Xs_out, model 

#################################################################################
## Builds


def b1_mlp(num_pixels, batch_size):
    model = Sequential()
    ### CHECKER # input shape
    model.add(Dense(num_pixels, input_shape=(num_pixels,), activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(batch_size*4, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(batch_size, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(batch_size/4, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(1, activation='sigmoid'))
    
    return model