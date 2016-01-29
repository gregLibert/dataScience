from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from matplotlib import pyplot
import lasagne
import numpy as np
import timeit

import data

print "Loading data"


X, y = data.load()

f = open('batchSize - executionTime','w')

print "Phase: full"

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
start_time = timeit.default_timer()

model_full = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_full.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_full.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_full.compile(loss='mean_squared_error', optimizer=sgd)

result_full = model_full.fit(X, y, nb_epoch=400, batch_size=16, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

duration_full = timeit.default_timer() - start_time

f.write('full: {}\n'.format(duration_full))

print result_full.history['loss']


print "Phase: 16"

early_stopping = EarlyStopping(monitor='val_loss', patience=100)
start_time = timeit.default_timer()

model_16 = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_16.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_16.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
model_16.compile(loss='mean_squared_error', optimizer=sgd)

result_16 = model_16.fit(X, y, nb_epoch=400, batch_size=16, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

duration_16 = timeit.default_timer() - start_time
f.write('16: {}\n'.format(duration_16))

print "Phase: 32"

early_stopping = EarlyStopping(monitor='val_loss', patience=100)
start_time = timeit.default_timer()

model_32 = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_32.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_32.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_32.compile(loss='mean_squared_error', optimizer=sgd)

result_32 = model_32.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

duration_32 = timeit.default_timer() - start_time
f.write('32: {}\n'.format(duration_32))

print "Phase: 64"

early_stopping = EarlyStopping(monitor='val_loss', patience=100)
start_time = timeit.default_timer()

model_64 = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_64.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_64.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_64.compile(loss='mean_squared_error', optimizer=sgd)

result_64 = model_64.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

duration_64 = timeit.default_timer() - start_time
f.write('64: {}\n'.format(duration_64))

print "Phase: 128"

early_stopping = EarlyStopping(monitor='val_loss', patience=100)
start_time = timeit.default_timer()

model_128 = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_128.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_128.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_128.compile(loss='mean_squared_error', optimizer=sgd)

result_128 = model_128.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

duration_128 = timeit.default_timer() - start_time
f.write('128: {}\n'.format(duration_128))

print "Phase: 256"

early_stopping = EarlyStopping(monitor='val_loss', patience=100)
start_time = timeit.default_timer()

model_256 = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_256.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_256.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_256.compile(loss='mean_squared_error', optimizer=sgd)

result_256 = model_256.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

duration_256 = timeit.default_timer() - start_time
f.write('256: {}\n'.format(duration_256))

print "Phase:512"

early_stopping = EarlyStopping(monitor='val_loss', patience=100)
start_time = timeit.default_timer()

model_512 = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_512.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_512.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_512.compile(loss='mean_squared_error', optimizer=sgd)

result_512 = model_512.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

duration_512 = timeit.default_timer() - start_time
f.write('512: {}\n'.format(duration_512))

print "Phase: 1024"

early_stopping = EarlyStopping(monitor='val_loss', patience=100)
start_time = timeit.default_timer()

model_1024 = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model_1024.add(Dense(100, input_dim=9216, init='glorot_uniform', activation='relu'))
model_1024.add(Dense(30, init='glorot_uniform', activation='linear'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_1024.compile(loss='mean_squared_error', optimizer=sgd)

result_1024 = model_1024.fit(X, y, nb_epoch=400, batch_size=1926, validation_split=0.2, show_accuracy=True, verbose=1, callbacks=[early_stopping])

duration_1024 = timeit.default_timer() - start_time
f.write('1024: {}\n'.format(duration_1024))

pyplot.plot(result_128.epoch * (duration_128/len(result_128.epoch) ), result_128.history['loss'], linestyle="dashed",  label="128")
pyplot.plot(result_32.epoch  * (duration_32/len(result_32.epoch)), result_32.history['loss'], linestyle="dashdot", label="32")
pyplot.plot(result_16.epoch  * (duration_16/len(result_16.epoch)), result_16.history['loss'], linestyle="dotted", label="16")
pyplot.plot(result_full.epoch  * (duration_full/len(result_full.epoch)), result_full.history['loss'], linestyle="dashed", label="1")
pyplot.plot(result_256.epoch  * (duration_256/len(result_256.epoch)), result_256.history['loss'], label="256")
pyplot.plot(result_64.epoch  * (duration_64/len(result_64.epoch)), result_64.history['loss'], linestyle="dashdot", label="64")
pyplot.plot(result_512.epoch  * (duration_512/len(result_512.epoch)), result_512.history['loss'], linestyle="dashdot", label="512")
pyplot.plot(result_1024.epoch  * (duration_1024/len(result_1025.epoch)), result_1024.history['loss'], linestyle="dotted", label="1024")


pyplot.grid()
pyplot.legend(loc='upper right')
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
#pyplot.ylim(1e-3, 1e-2)
pyplot.yscale("log")
pyplot.savefig('batch size - simple fully connected layer.png', bbox_inches='tight')