from keras.layers import Dense
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras_drop_block import DropBlock2D
from tensorflow.keras import layers, models

""" ENVIRONMENT VARIABLES """
size_filters = (5, 5)
size_filters_2 = (3, 3)
number_filters = 60
number_filters_2 = 32
size_of_pool = (2, 2)
number_nodes = 500

"""
Creates type B blocks as in Architecture.png
Parameters:
    cnn:            CNN model which the block belongs to
    n_filters:      Number of filters in the block
    kernel_conv1:   Dimension of kernel in the first Convolution layer
    kernel_conv2:   Dimension of kernel in the second Convolution layer
    kernel_mp:      Dimension of kernel in the MaxPooling layer, if present
    mp_stride:      Dimension of stride in the MaxPooling layer, if present
    kernel_db:      Dimension of kernel in the DropBlock layer
    db_percent:     Keep probability of the DropBlock layer
    input_shape:    Shape of the input for the first layer, if present
    name:           Name of the block, which will be in all layers names
"""


def create_b_block(cnn, n_filters, kernel_conv1, kernel_conv2, kernel_mp, mp_stride, kernel_db, db_percent, input_shape,
                   name):
    if input_shape:
        cnn.add(layers.Conv2D(n_filters, kernel_conv1, activation='relu', name=name + '_Conv1', padding='same',
                              input_shape=input_shape))
    else:
        cnn.add(layers.Conv2D(n_filters, kernel_conv1, activation='relu', name=name + '_Conv1', padding='same'))
    cnn.add(layers.BatchNormalization(name=name + '_BN1'))
    cnn.add(layers.Conv2D(n_filters, kernel_conv2, activation='relu', name=name + '_Conv2', padding='same'))
    cnn.add(layers.BatchNormalization(name=name + '_BN2'))
    if kernel_mp and mp_stride:
        cnn.add(layers.MaxPooling2D(kernel_mp, strides=mp_stride, name=name + '_MP1'))
    cnn.add(DropBlock2D(block_size=kernel_db, keep_prob=db_percent, name=name + '_DB1'))


"""
Creates type C blocks as in Architecture.png
Parameters:
    cnn:        CNN model which the block belongs to
    n_neurons1: Number of neurons in the first Fully Connected layer
    rate:       Rate in the DropOut layer
    n_neurons2: Number of neurons in the second Fully Connected layer
    name:       Name of the block, which will also be in all layers names
"""


def create_c_block(cnn, n_neurons1, rate, n_neurons2, name):
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(n_neurons1, name=name + '_FC_ReLu'))
    cnn.add(layers.Dropout(rate, name=name + '_DO1'))
    cnn.add(layers.Dense(n_neurons2, activation='softmax', name=name + '_FC_Softmax'))


"""
Creates the models of Class_CNN and Cat_CNN and prints their summaries
"""


def create_cnns(input_shape, number_classes):
    # Class_CNN creation
    class_cnn = models.Sequential()
    class_cnn._name = 'Class_CNN'

    # B1 Class
    create_b_block(class_cnn, number_filters_2, size_filters_2, size_filters_2, size_of_pool, 2, 3, 0.8, input_shape,
                   'B1')

    # B2 Class
    create_b_block(class_cnn, number_filters_2 * 2, size_filters_2, size_filters_2, size_of_pool, 2, 3, 0.8, None, 'B2')

    # B3 Class
    create_b_block(class_cnn, number_filters_2 * 4, size_filters_2, size_filters_2, size_of_pool, 2, 3, 0.75, None,
                   'B3')

    # B4 Class
    create_b_block(class_cnn, number_filters_2 * 8, (5, 1), (1, 5), None, None, 3, 0.75, None, 'B4')

    # B5 Class
    create_b_block(class_cnn, number_filters_2 * 10, (3, 1), (1, 3), size_of_pool, 2, 3, 0.75, None, 'B5')

    # C Class
    create_c_block(class_cnn, number_filters_2 * 8, 0.5, 43, 'C')

    # Class_CNN_2 creation
    class_cnn_2 = models.Sequential()
    class_cnn_2._name = 'Class_CNN_2'

    class_cnn_2.add((Conv2D(number_filters, size_filters, input_shape=input_shape, activation='relu')))
    class_cnn_2.add((Conv2D(number_filters, size_filters, activation='relu')))
    class_cnn_2.add(MaxPooling2D(pool_size=size_of_pool))

    class_cnn_2.add((Conv2D(number_filters // 2, size_filters_2, activation='relu')))
    class_cnn_2.add((Conv2D(number_filters // 2, size_filters_2, activation='relu')))
    class_cnn_2.add(MaxPooling2D(pool_size=size_of_pool))
    class_cnn_2.add(Dropout(0.5))

    class_cnn_2.add(Flatten())
    class_cnn_2.add(Dense(number_nodes, activation='relu'))
    class_cnn_2.add(Dropout(0.5))
    class_cnn_2.add(Dense(number_classes, activation='softmax'))

    # Cat_CNN creation
    cat_cnn = models.Sequential()
    cat_cnn._name = 'Cat_CNN'

    # B1 Cat
    create_b_block(cat_cnn, number_filters_2, size_filters_2, size_filters_2, size_of_pool, 2, 3, 0.8, input_shape,
                   'B1')

    # B2 Cat
    create_b_block(cat_cnn, number_filters_2 * 2, size_filters_2, size_filters_2, size_of_pool, 2, 3, 0.8, None, 'B2')

    # B3 Cat
    cat_cnn.add(layers.Conv2D(80, size_filters_2, activation='relu', padding='same', name='B3_Conv1'))
    cat_cnn.add(layers.BatchNormalization(name='B3_BN1'))
    cat_cnn.add(layers.MaxPooling2D(size_of_pool, strides=2, name='B3_MP1'))
    cat_cnn.add(DropBlock2D(block_size=size_filters_2, keep_prob=0.75, name='B3_DB1'))

    # C Cat
    cat_cnn.add(layers.Dense(20, name='C_FC_ReLu'))
    cat_cnn.add(layers.Dropout(0.5, name='C_DO1'))
    cat_cnn.add(layers.Dense(5, activation='softmax', name='C_FC_Softmax'))

    return cat_cnn, class_cnn, class_cnn_2
