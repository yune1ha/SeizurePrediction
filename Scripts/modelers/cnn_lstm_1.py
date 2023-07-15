'''
abhijithrb 
https://github.com/abhijithrb/SeizurePrediction

'''


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
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

    input_shape = (X_train.shape[1], X_train.shape[2])
    print('input shape:', input_shape)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1]).astype('float32')
    X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[2], X_valid.shape[1]).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1]).astype('float32')

    input_shape = (X_train.shape[1], X_train.shape[2])
    print('output shape:', input_shape)
    
    Xs_out = (X_train, X_valid, X_test)
    return Xs_out, input_shape

####################################################################################
def cnn_lstm_1(Xs, learning_rate, batch_size):
    print('Initiated cnn1_lstm_1.py...')

    # plastic surgery
    Xs_out, input_shape = shaper(Xs)  
    
    # choose model                             ### CHECKER
    model = build3(input_shape, batch_size)
    
    opt_adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=opt_adam, 
                  metrics=['binary_accuracy'],
                 )
    
    return Xs_out, model


####################################################################################
## Builds

# def build2(input_shape, batch_size):
#     print('FINAL INPUT SHAPE:', input_shape)
#     with tf.name_scope('CNN_LSTM'):
#         model = Sequential()

#         with tf.name_scope('Conv1'):
#             model.add(Convolution2D(16, (5, 5), padding = 'same', strides = (2, 2)),
#                                       input_shape = (21,3000,8), name = 'Conv1'))

#         model.add(BatchNormalization())
#         model.add(Activation('relu'))

#         with tf.name_scope('Conv2'):
#             model.add(Convolution2D(32, (5, 5), padding = 'same', strides = (1, 1), name = 'Conv2')))
#             model.add(Activation('relu'))

#         with tf.name_scope('Pooling'):
#             model.add(MaxPooling2D(pool_size = (2, 2))))

#         with tf.name_scope('Conv3'):
#             model.add(Convolution2D(32, (5, 5), padding = 'same', strides = (1, 1), name = 'Conv3')))
#             model.add(Activation('relu'))

#         with tf.name_scope('Conv4'):
#             model.add(Convolution2D(32, (5, 5), padding = 'same', strides = (1, 1), name = 'Conv4')))
#             model.add(Activation('relu'))

#         with tf.name_scope('Pooling'):
#             model.add(MaxPooling2D(pool_size = (2, 2))))

#         with tf.name_scope('FC1'):
#             model.add(Flatten(), name = 'FC1'))
#             model.add(Activation('relu'))

#             model.add(Dropout(0.25)))

#         with tf.name_scope('FC2'):
#             model.add(Dense(256), name = 'FC2'))
#             model.add(Activation('relu'))

#             model.add(Dropout(0.25)))

#         with tf.name_scope('LSTM'):
#             model.add(tf.keras.layers.LSTM(64)) #, return_sequences = False
#             model.add(Dropout(0.5))

#         with tf.name_scope('OutputLayer'):
#             model.add(Dense(2, activation = 'softmax'))

#     with tf.name_scope('Optimizer'):
#         optimizer = tf.keras.optimizers.Adam(lr = 1e-4, decay = 1e-5)

#     with tf.name_scope('Loss'):
#         model.compile(loss = 'categorical_crossentropy',
#                       optimizer = optimizer,
#                       metrics = ['accuracy'])

#     return model

# def build1(input_shape, batch_size):
#     ## Model Construction
#     model = Sequential()

#     # Reshape for TimeDistributed
#     '''
#     Note on TimeDistributed layer and input shape:
#     https://keras.io/api/layers/recurrent_layers/time_distributed/
#     '''
#     model.add(Reshape((3000, 21, 1), input_shape=(3000,21)))

#     # CNN
#     model.add(
#         Conv1D(
#             batch_size*2**0, kernel_size=2, activation='relu', padding='same', strides=2, 
#             input_shape=input_shape)#,
#         #input_shape=input_shape
#         ))
#     model.add(BatchNormalization())
    
#     model.add(Conv1D(
#         batch_size*2**1, kernel_size=2, activation='relu', padding='same', strides=1)))

#     # model.add(MaxPooling1D(2, padding='same')))

#     # model.add(Conv1D(
#     #     batch_size*2**1, kernel_size=2, activation='relu', padding='same', strides=1)))
    
#     # model.add(Conv1D(
#     #     batch_size*2**1, kernel_size=2, activation='relu', padding='same', strides=1)))

#     model.add(MaxPooling1D(2, padding='same')))

#     model.add(Flatten()))
#     model.add(Dropout(0.25)))

#     model.add(Dense(256, activation='relu')))
#     model.add(Dropout(0.25)))

#     # LSTM
#     model.add(Bidirectional(LSTM(batch_size*2**2, return_sequences=True)))
#     model.add(Dropout(0.5))
    
#     # Output
#     model.add(Dense(1, activation='sigmoid'))

#     return model
    
def build3(input_shape, batch_size):
    ## Model Construction
    model = Sequential()

    model.add(Reshape((3000, 21), input_shape=(3000,21)))

    # CNN
    model.add(
        Conv1D(
            batch_size*2**0, kernel_size=2, activation='relu', padding='same', 
            input_shape=input_shape),
        )
    model.add(BatchNormalization())
    
    model.add(Conv1D(
        batch_size*2**1, kernel_size=2, activation='relu', padding='same'))

    # model.add(BatchNormalization())
    # model.add(MaxPooling1D(2, padding='same'))

    # model.add(Conv1D(
    #     batch_size*2**1, kernel_size=2, activation='relu', padding='same'))
    
    # model.add(Conv1D(
    #     batch_size*2**1, kernel_size=2, activation='relu', padding='same'))
    
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2, padding='same'))

    # model.add(Flatten())
    # model.add(Dropout(0.25))

    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.25))

    # LSTM
    # model.add(Bidirectional(LSTM(
    #     64, dropout=0.1, recurrent_dropout=0.5, return_sequences=True)))
    model.add(Bidirectional(LSTM(
        64)))
    model.add(Dropout(0.5))
    
    
    # Output
    model.add(Dense(1, activation='sigmoid'))

    return model