from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from matplotlib import pyplot
import lasagne
import numpy as np

import data

print "Loading data"


X, y = data.load()

print "Phase: SGD_nesterov"

model_SGD_nesterov = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_SGD_nesterov.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_SGD_nesterov.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_SGD_nesterov.compile(loss='mean_squared_error', optimizer=sgd)

result_SGD_nesterov = model_SGD_nesterov.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1)

print "Phase: SGD"

model_SGD = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_SGD.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_SGD.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
model_SGD.compile(loss='mean_squared_error', optimizer=sgd)

result_SGD = model_SGD.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1)

print "Phase: RMSprop"

model_RMSprop = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_RMSprop.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_RMSprop.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = RMSprop()
model_RMSprop.compile(loss='mean_squared_error', optimizer=sgd)

result_RMSprop = model_RMSprop.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1)

print "Phase: Adagrad"

model_Adagrad = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_Adagrad.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_Adagrad.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = Adagrad()
model_Adagrad.compile(loss='mean_squared_error', optimizer=sgd)

result_Adagrad = model_Adagrad.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1)

print "Phase: Adadelta"

model_Adadelta = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_Adadelta.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_Adadelta.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = Adadelta()
model_Adadelta.compile(loss='mean_squared_error', optimizer=sgd)

result_Adadelta = model_Adadelta.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1)

print "Phase: Adam"

model_Adam = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_Adam.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_Adam.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = Adam()
model_Adam.compile(loss='mean_squared_error', optimizer=sgd)

result_Adam = model_Adam.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1)

print "Phase: Adamax"

model_Adamax = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_Adamax.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_Adamax.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = Adamax()
model_Adamax.compile(loss='mean_squared_error', optimizer=sgd)

result_Adamax = model_Adamax.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1)


pyplot.plot(result_SGD_nesterov.epoch, result_SGD_nesterov.history['acc'], label="Acc")
pyplot.plot(result_SGD.epoch, result_SGD.history['acc'], label="Acc")
pyplot.plot(result_RMSprop.epoch, result_RMSprop.history['acc'], label="Acc")
pyplot.plot(result_Adagrad.epoch, result_Adagrad.history['acc'], label="Acc")
pyplot.plot(result_Adadelta.epoch, result_Adadelta.history['acc'], label="Acc")
pyplot.plot(result_Adam.epoch, result_Adam.history['acc'], label="Acc")
pyplot.plot(result_Adamax.epoch, result_Adamax.history['acc'], label="Acc")


pyplot.grid()
pyplot.legend(loc='lower left')
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
#pyplot.ylim(1e-3, 1e-2)
pyplot.yscale("log")
pyplot.savefig('optimizers - simple fully connected layer.png', bbox_inches='tight')

print history.history