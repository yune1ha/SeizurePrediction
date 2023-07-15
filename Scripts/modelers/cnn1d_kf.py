import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, BatchNormalization, 
    Conv1D, MaxPooling1D, GlobalAveragePooling1D)
from sklearn.preprocessing import MinMaxScaler

# Settings
kernel_size = 2
scaler = MinMaxScaler()

# Knife
def shaper(Xs):
    X_train, X_test = Xs
    print('in Xs:', X_train.shape, X_test.shape)
    
    # Scale (for 3D shaped X)            
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    input_shape = (X_train.shape[1], X_train.shape[2])
    print('input shape:', input_shape)

    X_train = X_train.reshape(X_train.shape).astype('float32')
    X_test = X_test.reshape(X_test.shape).astype('float32')
    
    Xs_out = (X_train, X_test)
    return Xs_out, input_shape

####################################################################################
def cnn1d(Xs, learning_rate, batch_size):
    print('Initiated cnn1d.py...')

    # plastic surgery
    Xs_out, input_shape = shaper(Xs)  
    
    # choose model                             ### CHECKER
    model = jana2021_1d(input_shape, batch_size)
    
    opt_adam = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=opt_adam, 
                  metrics=['binary_accuracy'],
                 )
    
    return Xs_out, model


####################################################################################
## Builds

# def build_1(input_shape, batch_size):
#     model = Sequential()
#     #C1
#     model.add(Conv1D(filters=batch_size*2**0, kernel_size=kernel_size, padding='valid', activation='relu', input_shape=input_shape))
#     model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
#     model.add(BatchNormalization())
    
#     #C2
#     model.add(Conv1D(filters=batch_size*2**1, kernel_size=kernel_size, padding='valid', activation='relu'))
#     model.add(MaxPooling1D(pool_size=2, strides=2))
#     model.add(BatchNormalization())
#     #C3
#     model.add(Conv1D(filters=batch_size*2**2, kernel_size=kernel_size, padding='valid', activation='relu'))
#     model.add(MaxPooling1D(pool_size=2, strides=1))
#     model.add(BatchNormalization())
    
#     model.add(Dropout(0.5))
#     model.add(GlobalAveragePooling1D())
#     model.add(Dense(1, activation='sigmoid')) 
#     # use 1,sigmoid (instead of 2,softmax) for binary classification
#     return model

def jana2021_1d(input_shape, batch_size):
    ## Model Construction
    model = Sequential()
    model.add(Conv1D(batch_size*2**0, kernel_size=kernel_size, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D((1)))
    model.add(BatchNormalization())
    
    model.add(Conv1D(batch_size*2**1, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(MaxPooling1D((2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv1D(batch_size*2**2, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(MaxPooling1D((2), padding='same'))
    model.add(BatchNormalization())
    
    model.add(Conv1D(batch_size*2**3, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(MaxPooling1D((2), padding='same'))
    model.add(BatchNormalization())
    
    model.add(Conv1D(batch_size*2**4, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(MaxPooling1D((2), padding='same'))
    model.add(BatchNormalization())
    
    model.add(Conv1D(batch_size*2**5, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(MaxPooling1D((1), padding='same'))
    model.add(BatchNormalization())
    
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid')) 
    # use 1,sigmoid (instead of 2,softmax) for binary classification
    
    return model


# def build_3(input_shape, batch_size): #single channel
#     model = Sequential()
#     #C1
#     model.add(Conv1D(filters=batch_size*2**0, kernel_size=kernel_size, activation='relu', padding='same', input_shape=input_shape))
#     model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
#     model.add(BatchNormalization())
    
#     #C2
#     model.add(Conv1D(filters=batch_size*2**1, kernel_size=kernel_size, activation='relu', padding='same'))
#     model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
#     model.add(BatchNormalization())
    
#     #C3
#     model.add(Conv1D(filters=batch_size*2**2, kernel_size=kernel_size, activation='relu', padding='same'))
#     model.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
#     model.add(BatchNormalization())
    
#     model.add(Dropout(0.5))
#     model.add(GlobalAveragePooling1D())
#     model.add(Dense(1, activation='sigmoid')) 
#     # use 1,sigmoid (instead of 2,softmax) for binary classification
#     return model


# def build_2(input_shape, batch_size):  
#     # https://github.com/LiAngelo/seizure_prediction_cnn/blob/main/models/CNN_1d.py
#     model = Sequential()
#     model.add(Conv1D(batch_size*2**0, 12, strides=4, activation='relu', padding="same", 
#                      input_shape=input_shape))
#     model.add(MaxPooling1D(pool_size=3, strides=2))
#     model.add(BatchNormalization())
    
#     model.add(Conv1D(batch_size*2**1, 4, activation='relu',padding="same"))
#     model.add(MaxPooling1D(pool_size=3, strides=2))
#     model.add(BatchNormalization())
#     model.add(Conv1D(batch_size*2**2, 2, activation='relu',padding="same"))
#     model.add(MaxPooling1D(pool_size=3, strides=2))
#     model.add(BatchNormalization())
    
#     model.add(GlobalAveragePooling1D())
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='sigmoid'))
    
#     return model