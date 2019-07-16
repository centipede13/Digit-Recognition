# ----------------------
# - Reading the input data:

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# Running The network

import network

net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

"""
    Epoch 0 : 9089 / 10000
    Epoch 1 : 9268 / 10000
    Epoch 2 : 9262 / 10000
    Epoch 3 : 9327 / 10000
    Epoch 4 : 9380 / 10000
    Epoch 5 : 9362 / 10000
    Epoch 6 : 9338 / 10000
    Epoch 7 : 9435 / 10000
    Epoch 8 : 9433 / 10000
    Epoch 9 : 9449 / 10000
    Epoch 10 : 9460 / 10000

.........

    Epoch 20 : 9464 / 10000
    Epoch 21 : 9479 / 10000
    Epoch 22 : 9505 / 10000
    Epoch 23 : 9480 / 10000
    Epoch 24 : 9500 / 10000
    Epoch 25 : 9487 / 10000
    Epoch 26 : 9494 / 10000
    Epoch 27 : 9500 / 10000
    Epoch 28 : 9505 / 10000
    Epoch 29 : 9519 / 10000

i.e. 95.19% at its peak at epoch 29

"""
