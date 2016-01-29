from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ThresholdedLinear, ThresholdedReLU
from keras.callbacks import EarlyStopping

import timeit

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from matplotlib import pyplot
import lasagne
import numpy as np

import data

print "Loading data"


X, y = data.load()

f = open('activationFunction - executionTime','w')


print "Phase: relu"

start_time = timeit.default_timer()
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model_relu = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_relu.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_relu.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_relu.compile(loss='mean_squared_error', optimizer=sgd)

result_relu = model_relu.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

f.write('rlue: {}\n'.format(timeit.default_timer() - start_time))

print "Phase: tanh"

start_time = timeit.default_timer()
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model_tanh = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_tanh.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='tanh'))
model_tanh.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
model_tanh.compile(loss='mean_squared_error', optimizer=sgd)

result_tanh = model_tanh.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

f.write('tanh: {}\n'.format(timeit.default_timer() - start_time))

print "Phase: sigmoid"

start_time = timeit.default_timer()
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model_sigmoid = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_sigmoid.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='sigmoid'))
model_sigmoid.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_sigmoid.compile(loss='mean_squared_error', optimizer=sgd)

result_sigmoid = model_sigmoid.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

f.write('sigmoid: {}\n'.format(timeit.default_timer() - start_time))

print "Phase: hard_sigmoid"

start_time = timeit.default_timer()
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model_hard_sigmoid = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_hard_sigmoid.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='hard_sigmoid'))
model_hard_sigmoid.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_hard_sigmoid.compile(loss='mean_squared_error', optimizer=sgd)

result_hard_sigmoid = model_hard_sigmoid.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

f.write('hard_sigmoid: {}\n'.format(timeit.default_timer() - start_time))

print "Phase: ELU"

start_time = timeit.default_timer()
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model_ELU = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_ELU.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='ELU'))
model_ELU.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_ELU.compile(loss='mean_squared_error', optimizer=sgd)

result_ELU = model_ELU.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

f.write('ELU: {}\n'.format(timeit.default_timer() - start_time))

print "Phase: ThresholdedLinear"

start_time = timeit.default_timer()

model_ThresholdedLinear = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_ThresholdedLinear.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='ThresholdedLinear'))
model_ThresholdedLinear.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_ThresholdedLinear.compile(loss='mean_squared_error', optimizer=sgd)

result_ThresholdedLinear = model_ThresholdedLinear.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

f.write('ThresholdedLinear: {}\n'.format(timeit.default_timer() - start_time))

print "Phase: ThresholdedReLU"

start_time = timeit.default_timer()

model_ThresholdedReLU = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_ThresholdedReLU.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='ThresholdedReLU'))
model_ThresholdedReLU.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_ThresholdedReLU.compile(loss='mean_squared_error', optimizer=sgd)

result_ThresholdedReLU = model_ThresholdedReLU.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

f.write('ThresholdedReLU: {}\n'.format(timeit.default_timer() - start_time))

print "Phase: linear"

start_time = timeit.default_timer()

model_linear = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_linear.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='linear'))
model_linear.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_linear.compile(loss='mean_squared_error', optimizer=sgd)

result_linear = model_linear.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

f.write('linear: {}\n'.format(timeit.default_timer() - start_time))

print "Phase: LeakyReLU"

start_time = timeit.default_timer()

model_LeakyReLU = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_LeakyReLU.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_LeakyReLU.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_LeakyReLU.compile(loss='mean_squared_error', optimizer=sgd)

result_LeakyReLU = model_LeakyReLU.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

f.write('LeakyReLU: {}\n'.format(timeit.default_timer() - start_time))

print "Phase: PReLU"

start_time = timeit.default_timer()

model_PReLU = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_PReLU.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_PReLU.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_PReLU.compile(loss='mean_squared_error', optimizer=sgd)

result_PReLU = model_PReLU.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

f.write('PReLU: {}\n'.format(timeit.default_timer() - start_time))

f.close()

pyplot.plot(result_ELU.epoch, result_ELU.history['acc'], label="ELU")
pyplot.plot(result_sigmoid.epoch, result_sigmoid.history['acc'], label="sigmoid")
pyplot.plot(result_tanh.epoch, result_tanh.history['acc'], label="tanh")
pyplot.plot(result_relu.epoch, result_relu.history['acc'], label="relu")
pyplot.plot(result_ThresholdedLinear.epoch, result_ThresholdedLinear.history['acc'], label="ThresholdedLinear")
pyplot.plot(result_hard_sigmoid.epoch, result_hard_sigmoid.history['acc'], label="hard_sigmoid")
pyplot.plot(result_ThresholdedReLU.epoch, result_ThresholdedReLU.history['acc'], label="ThresholdedReLU")
pyplot.plot(result_LeakyReLU.epoch, result_LeakyReLU.history['acc'], label="LeakyReLU")
pyplot.plot(result_PReLU.epoch, result_PReLU.history['acc'], label="PReLU")
pyplot.plot(result_linear.epoch, result_linear.history['acc'], label="linear")


pyplot.grid()
pyplot.legend(loc='upper right')
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
#pyplot.ylim(1e-3, 1e-2)
pyplot.yscale("log")
pyplot.savefig('activation function - simple fully connected layer.png', bbox_inches='tight')