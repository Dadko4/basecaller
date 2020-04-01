import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import collections

from keras.models import Sequential, Model
from keras.callbacks import Callback
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras.activations import relu
from keras.metrics import categorical_accuracy, mean_squared_error
from keras.callbacks import BaseLogger, ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras import backend as K
from keras.initializers import Ones, Zeros, glorot_normal
from CTCModel import CTCModel

import tensorflow as tf
import h5py
import numpy as np
from chiron_input import read_raw_data_sets, padding
# tf.compat.v1.enable_eager_execution()

def plot_signal(signal):
    X = [i for i in range(len(signal))]
    plt.plot(X, signal)
    plt.show()

train_ds = read_raw_data_sets(r"C:\Users\dadom\Desktop\Chiron-master\train\\",
                              max_segments_num=100000)
eval_ds = read_raw_data_sets(r"C:\Users\dadom\Desktop\Chiron-master\eval\\",
                             max_segments_num=1000)

K.clear_session()


def prepare_data(batch_size=50, train=True):
    while True:
        ds = train_ds if train else eval_ds
        batch = ds.next_batch(batch_size)
        X = batch[0].reshape((batch_size, 300, 1))
        X_lens = batch[1]
        y_vals = []
        y_lens = []
        for sample in batch[2]:
            padding(sample[0], X[0].shape[0]//3)
            y_vals.append(sample[0])
            y_lens.append(sample[1])
        y = np.array(y_vals)
        y_lens = np.array(y_lens)
        yield [X, y, X_lens, y_lens], np.zeros(len(X))


def clipped_relu(value):
    return K.relu(value, max_value=20)


def create_network(features, labels, padding_value, units=512, output_dim=5, 
                   dropout=0.2):
    input_data = Input(name='input', shape=(None, features))

    masking = Masking(mask_value=padding_value)(input_data)
    noise = GaussianNoise(0.01)(input_data)
    
    cnn0 = Conv1D(filters=units, kernel_size=5, strides=1, activation=clipped_relu,
               kernel_initializer='glorot_uniform', bias_initializer='random_normal',
               name='conv_1')(input_data)
    dropout0 = TimeDistributed(Dropout(dropout), name='dropout_1')(cnn0)

    cnn1 = Conv1D(filters=units, kernel_size=5, strides=1, activation=clipped_relu,
               kernel_initializer='glorot_uniform', bias_initializer='random_normal',
               name='conv_2')(dropout0)
    dropout1 = TimeDistributed(Dropout(dropout), name='dropout_2')(cnn1)

    cnn2 = Conv1D(filters=units, kernel_size=5, strides=2, activation=clipped_relu,
               kernel_initializer='glorot_uniform', bias_initializer='random_normal',
               name='conv_3')(dropout1)
    dropout2 = TimeDistributed(Dropout(dropout), name='dropout_3')(cnn2)

    blstm0 = Bidirectional(LSTM(units, return_sequences=True, dropout=dropout))(noise)
    blstm1 = Bidirectional(LSTM(units, return_sequences=True, dropout=dropout))(blstm0)
    blstm2 = Bidirectional(LSTM(units, return_sequences=True, dropout=dropout))(blstm1)

    dense0 = TimeDistributed(Dense(units=units, kernel_initializer='random_normal',
                                   bias_initializer='random_normal',activation='relu'),
                                   name='fc_4')(blstm1)
    dense1 = TimeDistributed(Dropout(dropout), name='dropout_4')(dense0)

    dense2 = TimeDistributed(Dense(labels + 1, name="dense"))(dense1)
    y_pred = Activation('softmax', name='softmax')(dense2)
    network = CTCModel([input_data], [y_pred])
    network.compile(Adam())
    return network


labels = 4
batch_size = 50
epochs = 10
features = 1
padding_value = 100

K.clear_session()
network = create_network(features, labels, padding_value)

X_val, y_val = prepare_data(10, train=False).__next__()
es = EarlyStopping(monitor='val_loss', mode='min')
# lr = LearningRateScheduler(schedule, verbose=0)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', 
                     verbose=1, save_best_only=True)
lr_cb = ReduceLROnPlateau(factor=0.2, patience=5, verbose=0, epsilon=0.1, 
                          min_lr=0.0000001)
network.fit_generator(prepare_data(batch_size), steps_per_epoch=2000, epochs=epochs,
                      validation_data=(X_val, y_val), callbacks=[es, mc, lr_cb])

pred = network.predict([X_val[0], X_val[2]], batch_size=batch_size, 
                        max_value=padding_value)
for i in range(10):
    print("Prediction :", [j for j in pred[i] if j!=-1])
