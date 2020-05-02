import numpy as np
# from matplotlib import pyplot as plt
from glob import glob
import collections

from keras.models import Sequential, Model
from keras.callbacks import Callback
from keras.layers import *
from keras.optimizers import Nadam, SGD
from keras.initializers import RandomNormal, glorot_uniform
from keras.activations import relu
from keras.metrics import categorical_accuracy, mean_squared_error
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.initializers import Ones, Zeros, glorot_normal
from CTCModel import CTCModel

import tensorflow as tf
import h5py
import numpy as np
from chiron_input import read_raw_data_sets, padding
from Bio.pairwise2 import align
import pickle

np.random.seed(1)
tf.set_random_seed(1)
train_ds = read_raw_data_sets(r"train/", max_segments_num=100000)
eval_ds = read_raw_data_sets(r"eval/", max_segments_num=20000)

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
        
        
def create_CNN_network(features, labels, padding_value, units=256, output_dim=5, 
                   dropout=0.2):
    initializer = RandomNormal(mean=0.0, stddev=0.05, seed=1)
    lstm_init = glorot_uniform(1)
    input_data = Input(name='input', shape=(None, features))

    # masking = Masking(mask_value=padding_value)(input_data)

    pad = ZeroPadding1D(padding=(0, 310))(input_data)

    cnn0 = Conv1D(filters=units, kernel_size=5, strides=1, activation='relu',
               kernel_initializer=lstm_init, bias_initializer=initializer,
               name='conv_1')(pad)
    dropout0 = TimeDistributed(Dropout(dropout), name='dropout_1')(cnn0)

    cnn1 = Conv1D(filters=units, kernel_size=5, strides=1, activation='relu',
               kernel_initializer=lstm_init, bias_initializer=initializer,
               name='conv_2')(dropout0)
    dropout1 = TimeDistributed(Dropout(dropout), name='dropout_2')(cnn1)

    cnn2 = Conv1D(filters=units, kernel_size=5, strides=2, activation='relu',
               kernel_initializer=lstm_init, bias_initializer=initializer,
               name='conv_3')(dropout1)
    dropout2 = TimeDistributed(Dropout(dropout), name='dropout_3')(cnn2)

    dense1 = TimeDistributed(Dense(units=units, kernel_initializer=initializer,
                                   bias_initializer=initializer, activation='relu'), 
                            name='fc_0')(dropout2)
    dropout3 = TimeDistributed(Dropout(dropout), name='dropout_4')(dense1)
    dense2 = TimeDistributed(Dense(units=units, kernel_initializer=initializer,
                                   bias_initializer=initializer, activation='relu'), 
                            name='fc_1')(dropout3)
    dropout4 = TimeDistributed(Dropout(dropout), name='dropout_5')(dense2)
    blstm0 = Bidirectional(LSTM(units, return_sequences=True, kernel_initializer=lstm_init, 
                                dropout=dropout))(dropout4)
    blstm1 = Bidirectional(LSTM(units, return_sequences=True, kernel_initializer=lstm_init, 
                                dropout=dropout))(blstm0)
    blstm2 = Bidirectional(LSTM(units, return_sequences=True, kernel_initializer=lstm_init, 
                                dropout=dropout))(blstm1)
    dense3 = TimeDistributed(Dense(units=units, kernel_initializer=initializer,
                                   bias_initializer=initializer,activation='relu'),
                                   name='fc_3')(blstm2)
    dropout4 = TimeDistributed(Dropout(dropout), name='dropout_6')(dense3)

    dense4 = TimeDistributed(Dense(labels + 1, name="dense"))(dropout4)
    y_pred = Activation('softmax', name='softmax')(dense4)
    network = CTCModel([input_data], [y_pred])
    network.compile(Nadam())
    return network


def create_LSTM_network(features, labels, padding_value, units=256, output_dim=5, 
                   dropout=0.2):
    initializer = RandomNormal(mean=0.0, stddev=0.05, seed=1)
    lstm_init = glorot_uniform(1)
    input_data = Input(name='input', shape=(None, features))

    masking = Masking(mask_value=padding_value)(input_data)

    # pad = ZeroPadding1D(padding=(0, 500))(input_data)
    noise = GaussianNoise(0.01)(masking)
    dense0 = TimeDistributed(Dense(units=units, kernel_initializer=initializer,
                                   bias_initializer=initializer, activation='relu'), 
                            name='fc_0')(noise)
    dropout0 = TimeDistributed(Dropout(dropout), name='dropout_0')(dense0)
    dense1 = TimeDistributed(Dense(units=units, kernel_initializer=initializer,
                                   bias_initializer=initializer, activation='relu'), 
                            name='fc_1')(dropout0)
    dropout1 = TimeDistributed(Dropout(dropout), name='dropout_1')(dense1)

    blstm0 = Bidirectional(LSTM(units, return_sequences=True, kernel_initializer=lstm_init, 
                                dropout=dropout))(dropout1)
    blstm1 = Bidirectional(LSTM(units, return_sequences=True, kernel_initializer=lstm_init, 
                                dropout=dropout))(blstm0)
    blstm2 = Bidirectional(LSTM(units, return_sequences=True, kernel_initializer=lstm_init, 
                                dropout=dropout))(blstm1)

    dense3 = TimeDistributed(Dense(units=units, kernel_initializer=initializer,
                                   bias_initializer=initializer,activation='relu'),
                                   name='fc_3')(blstm2)
    dropout3 = TimeDistributed(Dropout(dropout), name='dropout_3')(dense3)

    dense4 = TimeDistributed(Dense(labels + 1, name="dense"))(dropout3)
    y_pred = Activation('softmax', name='softmax')(dense4)
    network = CTCModel([input_data], [y_pred])
    network.compile(Nadam())
    return network


labels = 4
batch_size = 100
epochs = 35
features = 1
padding_value = 100

K.clear_session()
network = create_CNN_network(features, labels, padding_value, units=200)
print(network.summary())
X_val, y_val = prepare_data(20000, train=False).__next__()

es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
lr_cb = ReduceLROnPlateau(factor=0.2, patience=2, verbose=0, min_lr=0.0001)
history = network.fit_generator(prepare_data(batch_size), steps_per_epoch=1000, 
                                epochs=epochs, validation_data=(X_val, y_val), 
                                callbacks=[es, lr_cb])

with open('CNN_history.pickle', 'wb') as handle:
    pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
network.save_model('CNN_model/')

pred = network.predict([X_val[0], X_val[2]], batch_size=batch_size, 
                        max_value=padding_value)
with open('pred.pickle', 'wb') as handle:
    pickle.dump(pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('X_val.pickle', 'wb') as handle:
    pickle.dump(X_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

try:
    network.model_train.save_weights('CNN_model/CNN_weights.h5')
except:
    pass

from Bio.pairwise2 import align
y_val_true = []
for y_valid, lenght in zip(X_val[1], X_val[3]):
    y_val_true.append(y_valid[:lenght])
    
accuracies = []
for y_true, y_pred in zip(y_val_true, pred):
    y_true = ''.join(y_true.astype(str))
    y_pred = ''.join(y_pred.astype('int64').astype(str))
    y_pred = y_pred.replace('-1', '')
    if not len(y_pred):
        accuracies.append(0)
        continue
    aligned = np.array(align.globalxx(y_true, y_pred))[:, [2, -1]].astype('float16')
    align_score_ix = aligned[:, 0].argmax()
    align_score = aligned[:, 0][align_score_ix]
    align_length = aligned[:, 1][align_score_ix]
    accuracies.append(align_score/align_length)

with open('CNN_accuracies.pickle', 'wb') as handle:
    pickle.dump(accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print(np.mean(accuracies))
