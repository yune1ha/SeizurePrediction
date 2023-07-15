import tensorflow as tf
from tensorflow.keras.metrics import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, BatchNormalization, 
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Activation)
from sklearn.preprocessing import MinMaxScaler

# Settings
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
            
def cnn2d(Xs, learning_rate, batch_size):  
    print('Initiated cnn2d.py...')
    
    # plastic surgery
    Xs_out, input_shape = shaper(Xs)    
    
    # pick 1 model
    ### CHECKER
    model = jana2021(input_shape, batch_size)
    #model = build_1(input_shape, batch_size)
    
    opt_adam = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=opt_adam, 
                  metrics=[
                      'binary_accuracy'],
                 )
    
    return Xs_out, model 

####################################################################################
## Builds

# def build_1(input_shape, batch_size):
#     ## Model Construction
#     model = Sequential()
#     model.add(Conv2D(batch_size*2**0, (1,3), padding='valid', activation='relu', input_shape=input_shape))
#     model.add(MaxPooling2D(pool_size=(2, 2),  padding='same'))
#     model.add(BatchNormalization())
    
#     model.add(Conv2D(batch_size*2**1, (1,3), padding='valid', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(BatchNormalization())
#     model.add(Conv2D(batch_size*2**2, (1,3), padding='valid', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(BatchNormalization())
    
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='sigmoid')) 
#     # use 1,sigmoid (instead of 2,softmax) for binary classification
    
#     return model

def jana2021(input_shape, batch_size):
    ## Model Construction
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((1, 3)))
    
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 3), padding='same'))

    model.add(Conv2D(48, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 3), padding='same'))
    
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 3), padding='same'))
    
    model.add(Conv2D(96, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 3), padding='same'))
    
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((1, 3), padding='same'))
    
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid')) 
    # use 1,sigmoid (instead of 2,softmax) for binary classification
    
    return model
