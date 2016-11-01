from huffmax import HuffmaxClassifier
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import time

# TODO : Make this work

batch_size = 32
input_dim = 100
nb_classes = 50000
nb_samples = 10000


X = np.random.random((nb_samples, input_dim))

model = Sequential()
model.add(Dense(nb_classes, input_dim=input_dim, activation='softmax'))
model.predict(X[:1])

start_time = time.time()
model.predict(X)
end_time = time.time()
print model.predict(X)

softmax_time = end_time - start_time


print "Starting to compile Huffmax model"
model = Sequential()
model.add(HuffmaxClassifier(nb_classes, input_dim=input_dim, verbose=True))
model.predict(X[:1])
print "Compile Huffmax done."

start_time = time.time()
model.predict(X)
end_time = time.time()
print model.predict(X)

huffmax_time = end_time - start_time

speedup = softmax_time / huffmax_time

print('Speedup = ' + str(speedup) + 'X')
