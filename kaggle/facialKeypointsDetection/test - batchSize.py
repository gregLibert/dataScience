from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from matplotlib import pyplot
import lasagne
import numpy as np

import data

print "Loading data"


X, y = data.load()


early_stopping = EarlyStopping(monitor='val_loss', patience=10)

print "Phase: full"

model_full = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_full.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_full.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_full.compile(loss='mean_squared_error', optimizer=sgd)

result_full = model_full.fit(X, y, nb_epoch=400, batch_size=1, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

print "Phase: 16"

model_16 = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_16.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='tanh'))
model_16.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
model_16.compile(loss='mean_squared_error', optimizer=sgd)

result_16 = model_16.fit(X, y, nb_epoch=400, batch_size=16, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

print "Phase: 32"

model_32 = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_32.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='sigmoid'))
model_32.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_32.compile(loss='mean_squared_error', optimizer=sgd)

result_32 = model_32.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

print "Phase: 64"

model_64 = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_64.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='hard_sigmoid'))
model_64.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_64.compile(loss='mean_squared_error', optimizer=sgd)

result_64 = model_64.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

print "Phase: 128"

model_128 = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_128.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='ELU'))
model_128.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_128.compile(loss='mean_squared_error', optimizer=sgd)

result_128 = model_128.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

print "Phase: 256"

model_256 = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_256.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='ThresholdedLinear'))
model_256.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_256.compile(loss='mean_squared_error', optimizer=sgd)

result_256 = model_256.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

print "Phase:512"

model_512 = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_512.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='ThresholdedReLU'))
model_512.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_512.compile(loss='mean_squared_error', optimizer=sgd)

result_512 = model_512.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

print "Phase: 1024"

model_1024 = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_1024.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='linear'))
model_1024.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_1024.compile(loss='mean_squared_error', optimizer=sgd)

result_1024 = model_1024.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

pyplot.plot(result_128.epoch, result_128.history['acc'], label="128")
pyplot.plot(result_32.epoch, result_32.history['acc'], label="32")
pyplot.plot(result_16.epoch, result_16.history['acc'], label="16")
pyplot.plot(result_full.epoch, result_full.history['acc'], label="1-Full")
pyplot.plot(result_256.epoch, result_256.history['acc'], label="256")
pyplot.plot(result_64.epoch, result_64.history['acc'], label="64")
pyplot.plot(result_512.epoch, result_512.history['acc'], label="512")
pyplot.plot(result_1024.epoch, result_1024.history['acc'], label="1024")


pyplot.grid()
pyplot.legend(loc='lower left')
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
#pyplot.ylim(1e-3, 1e-2)
pyplot.yscale("log")
pyplot.savefig('batch size - simple fully connected layer.png', bbox_inches='tight')

print history.history