import numpy as np
# from matplotlib import pyplot as plt
from glob import glob
import collections

from keras.models import Sequential, Model
from keras.callbacks import Callback
from keras.layers import *
from keras.optimizers import Nadam, SGD
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
from Bio.pairwise2.align import globalxx
import pickle

# tf.compat.v1.enable_eager_execution()

# def plot_signal(signal):\
#     X = [i for i in range(len(signal))]
#     plt.plot(X, signal)
#     plt.show()

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

    # pad = ZeroPadding1D(padding=(0, 500))(input_data)
    noise = GaussianNoise(0.01)(masking)
    dense0 = TimeDistributed(Dense(units=units, kernel_initializer='random_normal',
                                   bias_initializer='random_normal', activation=clipped_relu), 
                            name='fc_0')(noise)
    dropout0 = TimeDistributed(Dropout(dropout), name='dropout_0')(dense0)
    dense1 = TimeDistributed(Dense(units=units, kernel_initializer='random_normal',
                                   bias_initializer='random_normal', activation=clipped_relu), 
                            name='fc_1')(dropout0)
    dropout1 = TimeDistributed(Dropout(dropout), name='dropout_1')(dense1)
    # dense2 = TimeDistributed(Dense(units=units, kernel_initializer='random_normal',
    #                                bias_initializer='random_normal', activation=clipped_relu), 
    #                         name='fc_2')(dropout1)
    # dropout2 = TimeDistributed(Dropout(dropout), name='dropout_2')(dense2)

    # cnn0 = Conv1D(filters=units, kernel_size=5, strides=1, activation=clipped_relu,
    #            kernel_initializer='glorot_uniform', bias_initializer='random_normal',
    #            name='conv_1')(emb)
    # dropout0 = TimeDistributed(Dropout(dropout), name='dropout_1')(cnn0)

    # cnn1 = Conv1D(filters=units, kernel_size=5, strides=1, activation=clipped_relu,
    #            kernel_initializer='glorot_uniform', bias_initializer='random_normal',
    #            name='conv_2')(dropout0)
    # dropout1 = TimeDistributed(Dropout(dropout), name='dropout_2')(cnn1)

    # cnn2 = Conv1D(filters=units, kernel_size=5, strides=2, activation=clipped_relu,
    #            kernel_initializer='glorot_uniform', bias_initializer='random_normal',
    #            name='conv_3')(dropout1)
    # dropout2 = TimeDistributed(Dropout(dropout), name='dropout_3')(cnn2)

    blstm0 = Bidirectional(LSTM(units, return_sequences=True, dropout=dropout))(dropout1)
    blstm1 = Bidirectional(LSTM(units, return_sequences=True, dropout=dropout))(blstm0)
    blstm2 = Bidirectional(LSTM(units, return_sequences=True, dropout=dropout))(blstm1)
    blstm3 = Bidirectional(LSTM(units, return_sequences=True, dropout=dropout))(blstm2)
    dropout2 = Bidirectional(LSTM(units, return_sequences=True, dropout=dropout))(blstm3)

    dense3 = TimeDistributed(Dense(units=units, kernel_initializer='random_normal',
                                   bias_initializer='random_normal',activation='relu'),
                                   name='fc_3')(dropout2)
    dropout3 = TimeDistributed(Dropout(dropout), name='dropout_3')(dense3)

    dense4 = TimeDistributed(Dense(labels + 1, name="dense"))(dropout3)
    y_pred = Activation('softmax', name='softmax')(dense4)
    network = CTCModel([input_data], [y_pred])
    network.compile(Nadam())
    return network


labels = 4
batch_size = 100
epochs = 10
features = 1
padding_value = 100

K.clear_session()
network = create_network(features, labels, padding_value)

X_val, y_val = prepare_data(10, train=False).__next__()

es = EarlyStopping(monitor='val_loss', mode='min', patience=3)
lr_cb = ReduceLROnPlateau(factor=0.2, patience=2, verbose=0, epsilon=0.1, 
                          min_lr=0.0000001)
network.fit_generator(prepare_data(batch_size), steps_per_epoch=1000, epochs=epochs,
                      validation_data=(X_val, y_val), callbacks=[es, lr_cb])

network.save_model('models/')

pred = network.predict([X_val[0], X_val[2]], batch_size=batch_size, 
                        max_value=padding_value)

accuracies = []
for y_true, y_pred in zip(y_val, pred):
    y_true = ''.join(y_true)
    y_pred = ''.join(y_pred)
    align = np.array(globalxx(y_true, y_pred))[:, [2, -1]].astype('float16')
    align_score_ix = align[:, 0].argmax()
    align_score = align[:, 0][align_score_ix]
    align_length = align[:, 1][align_score_ix]
    accuracies.append(align_score/align_length)

with open('y_pred.pickle', 'wb') as handle:
    pickle.dump(y_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('y_true.pickle', 'wb') as handle:
    pickle.dump(y_true, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('accuracies.pickle', 'wb') as handle:
    pickle.dump(accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(np.mean(accuracies))
