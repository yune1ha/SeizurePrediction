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

    X_train = X_train.astype('float32').reshape((X_train.shape[0], X_train.shape[2], X_train.shape[1]))
    X_valid = X_valid.astype('float32').reshape(X_valid.shape[0], X_valid.shape[2], X_valid.shape[1])
    X_test = X_test.astype('float32').reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])

    input_shape = (X_train.shape[1], X_train.shape[2])
    print('output shape:', input_shape)
    
    Xs_out = (X_train, X_valid, X_test)
    return Xs_out, input_shape

####################################################################################
def lstm_1(Xs, learning_rate, batch_size):
    print('Initiated lstm_1.py...')

    # plastic surgery
    Xs_out, input_shape = shaper(Xs)  
    
    # choose model                             ### CHECKER
    model = build1(input_shape, batch_size)
    
    opt_adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=opt_adam, 
                  metrics=['binary_accuracy'],
                 )
    
    return Xs_out, model

def build1(input_shape, batch_size):
    model = Sequential()

    # LSTM recommended time step around 1000

    #model.add(Reshape((3000, 1), input_shape=(3000, 21)))

    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    #model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    #model.add(Softmax())
    model.add(Dense(1, activation='sigmoid'))

    return model