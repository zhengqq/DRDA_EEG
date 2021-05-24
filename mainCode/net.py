import tensorflow as tf
import scipy.io as sio
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow import keras




def lrelu(x, leak=0.05, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def build_model(size_y, size_x, dim=1):
    # input layer
    img_input = keras.layers.Input(shape=(size_y, size_x, dim))

    # First convolution extracts 30 filters that are (size_y, 3)
    # Convolution is followed by max-pooling layer with a 1x10 window
    x = keras.layers.Conv2D(filters=30, kernel_size=(size_y, 3), activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0))(img_input)
    x = keras.layers.MaxPooling2D(1, 10)(x)

    # Convolution is followed by max-pooling layer with a 1x10 window
    x = keras.layers.Conv2D(filters=10, kernel_size=(1, 5), activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.05))(x)
    x = keras.layers.MaxPooling2D(1, 3)(x)

    # Flatten feature map to a 1-dim tensor so we can add fully connected layers
    x = keras.layers.Flatten()(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    # x = keras.layers.MaxPooling1D(2)(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    # x = keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)

    # Add a dropout rate of 0.5
    # x = keras.layers.Dropout(0.75)(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    # x = keras.layers.Dense(256, activation='sigmoid')(x)

    # Add a dropout rate of 0.5
    # x = keras.layers.Dropout(0.75)(x)

    # Create output layer with a single node and sigmoid activation
    output = keras.layers.Dense(2, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0))(x)

    # Create model:
    # input = input feature map
    # output = input feature map + stacked convolution/maxpooling layers + fully
    # connected layer + sigmoid output layer
    model = keras.models.Model(img_input, output)
    model.summary()
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=keras.optimizers.RMSprop(lr=0.001),
    #               metrics=['acc'])
    op1=keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=op1,
                  metrics=['acc'])
    return model














def spectrogram_net(input, channel_size, cls_num=1):
    # weights_regularizer = slim.l1_regularizer(0.01)
    weights_regularizer = slim.l1_regularizer(0.00001) # best 0.00001
    # weights_regularizer = None
    bn = slim.batch_norm

    net = slim.conv2d(input, 16, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.conv2d(net, 16, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.conv2d(net, 16, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.max_pool2d(net,[1,2], stride=[1,2])

    net = slim.conv2d(net, 32, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.conv2d(net, 32, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.conv2d(net, 32, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.max_pool2d(net,[1,2], stride=[1,2])

    net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.max_pool2d(net,[1,7], stride=[1,7])


    fc = slim.flatten(net)

    net = slim.fully_connected(fc, 64, weights_regularizer=None)
    # net = slim.dropout(net)
    net = slim.fully_connected(net, 32, weights_regularizer=None)
    # net = slim.dropout(net)
    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)

    return net, pred

def spectrogram_net2(input, channel_size, cls_num=1):
    # weights_regularizer = slim.l1_regularizer(0.01)
    weights_regularizer = slim.l1_regularizer(0.00001) # best 0.00001
    # weights_regularizer = None

    net = slim.conv2d(input, 16, [1, 10], weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 16, [1, 10],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 16, [1, 3],weights_regularizer = weights_regularizer)
    net = slim.max_pool2d(net,[1,2], stride=[1,2])

    net = slim.conv2d(net, 32, [1, 5],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 32, [1, 5],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 32, [1, 3],weights_regularizer = weights_regularizer)
    net = slim.max_pool2d(net,[1,2], stride=[1,2])

    net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer)
    net = slim.max_pool2d(net,[1,2], stride=[1,2])


    fc = slim.flatten(net)

    net = slim.fully_connected(fc, 128, weights_regularizer=None)
    # net = slim.dropout(net)
    net = slim.fully_connected(net, 64, weights_regularizer=None)
    # net = slim.dropout(net)
    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)

    return net, pred

def resnet(input, channel_size, cls_num=1):
    weights_regularizer = slim.l1_regularizer(0.00001)  # best 0.00001

    net = slim.conv2d(input, 64, [7,7],stride=2, weights_regularizer=weights_regularizer)
    net = slim.max_pool2d(net,[3,3],stride=2)

    x = net
    net = slim.conv2d(net, 64, [3, 3], normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = slim.conv2d(net, 64, [3, 3], normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = net + x

    x = slim.conv2d(net, 128, [1,1], stride=2, normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = slim.conv2d(net, 128, [3, 3], stride=2, normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = slim.conv2d(net, 128, [3, 3], normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = net + x

    x = slim.conv2d(net, 256, [1,1], stride=2, normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = slim.conv2d(net, 256, [3, 3], stride=2, normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = slim.conv2d(net, 256, [3, 3], normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = net + x

    net = slim.avg_pool2d(net, [2,2])

    net = slim.flatten(net)

    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)


    return net, pred







def spectrogram_net_tst(input, channel_size, cls_num=1):
    # weights_regularizer = slim.l1_regularizer(0.01)
    weights_regularizer = slim.l2_regularizer(0.00001) # best

    net = slim.conv2d(input, 16, [3, 3],weights_regularizer = weights_regularizer)
    net = slim.conv2d(net, 16, [3, 3],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 16, [1, 3],weights_regularizer = weights_regularizer)
    net = slim.max_pool2d(net,[2,2], stride=[2,2])

    net = slim.conv2d(net, 32, [3, 3],weights_regularizer = weights_regularizer)
    net = slim.conv2d(net, 32, [3, 3],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 32, [1, 3],weights_regularizer = weights_regularizer)
    net = slim.max_pool2d(net,[2,2], stride=[2,2])

    # net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer)
    # net = slim.max_pool2d(net,[1,7], stride=[1,7])


    fc = slim.flatten(net)

    net = slim.fully_connected(fc, 64, weights_regularizer=weights_regularizer)
    net = slim.fully_connected(net, 32, weights_regularizer=weights_regularizer)
    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)

    return net, pred


def spectrogram_net_1d(input, channel_size, cls_num):

    weights_regularizer = None#slim.l2_regularizer(0.01)

    net = slim.conv2d(input, 30, [channel_size, 3], padding='VALID', weights_regularizer=weights_regularizer)
    # net = slim.conv2d(net, 16, [1, 3], padding='SAME')
    net = slim.max_pool2d(net, [1, 10],stride=[1,10])

    net = slim.conv2d(net, 10, [1, 5], padding='VALID',weights_regularizer=weights_regularizer)
    # net = slim.conv2d(net, 32, [1, 3], padding='SAME')
    net = slim.max_pool2d(net, [1, 2], stride=[1,2])

    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])

    fc = slim.flatten(net)

    net = slim.fully_connected(fc, 64, weights_regularizer=weights_regularizer)
    # net = slim.fully_connected(net, 32)
    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)

    return net, pred


def tju_csp(input,channel_size, cls_num=1):
    net = slim.conv2d(input, 32, [1, 4], padding='VALID', normalizer_fn=slim.batch_norm)

    net = slim.conv2d(net, 7, [7, 1], padding='VALID', normalizer_fn=slim.batch_norm)

    net = slim.avg_pool2d(net, [4,1], stride=[4,1],padding='SAME')

    net = slim.flatten(net)

    net = slim.fully_connected(net, 2, activation_fn=None)

    pred = tf.nn.softmax(net)

    return net, pred


def signal(input, channel_size, cls_num, is_training):

    weights_regularizer = slim.l2_regularizer(0.01)

    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None)
    # net = slim.conv2d(net, 16, [1, 3], padding='SAME')
    # net = slim.max_pool2d(net, [1, 10],stride=[1,10])

    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm)
    # net = slim.conv2d(net, 32, [1, 3], padding='SAME')
    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])

    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])

    fc = slim.flatten(net)

    net = slim.fully_connected(fc, 64, weights_regularizer=None)
    #net = slim.fully_connected(net, 32)
    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)

    return net, pred



def signal_siamese(input, channel_size, cls_num, is_training, reuse=False):



    weights_regularizer = slim.l2_regularizer(0.01)
    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None, reuse=reuse, scope='conv_time')
    # net = slim.conv2d(net, 16, [1, 3], padding='SAME')
    # net = slim.max_pool2d(net, [1, 10],stride=[1,10])
    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm, reuse=reuse, scope='conv_spatial')
    # net = slim.conv2d(net, 32, [1, 3], padding='SAME')
    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])
    fc = slim.flatten(net)
    net1 = slim.fully_connected(fc, 64, weights_regularizer=None, reuse=reuse, scope='fc')
    # net3 = slim.fully_connected(net1, 32, reuse=reuse, scope='fc1', activation_fn=tf.nn.relu)
    # net4 = slim.fully_connected(net3, 32, reuse=reuse, scope='fc2', activation_fn=tf.nn.leaky_relu)
    net = slim.fully_connected(net1, cls_num, activation_fn=None, reuse=reuse, scope='output')
    pred = tf.nn.softmax(net)
    # net4 = slim.fully_connected(net1, 8, reuse=reuse, scope='fea')

    return net, pred, net1


def signal_siamese_da(input, channel_size, cls_num, is_training=True, reuse=False):



    weights_regularizer = slim.l2_regularizer(0.01)
    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None, reuse=reuse, scope='g_conv_time')
    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm, reuse=reuse, scope='g_conv_spatial')

    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])
    fc = slim.flatten(net)

    net1 = slim.fully_connected(fc, 64, weights_regularizer=None, reuse=reuse, scope='g_fc')
    net2 = slim.fully_connected(net1, 32, reuse=reuse, scope='g_fc1')
    # net4 = slim.fully_connected(net3, 32, reuse=reuse, scope='fc2', activation_fn=tf.nn.leaky_relu)
    net = slim.fully_connected(net2, cls_num, activation_fn=None, reuse=reuse, scope='g_output')
    pred = tf.nn.softmax(net)
    # net4 = slim.fully_connected(net1, 8, reuse=reuse, scope='fea')

    return net, pred, net1



def signal_siamese_da_fc64(input, channel_size, cls_num, is_training=True, reuse=False):



    weights_regularizer = slim.l2_regularizer(0.01)
    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None, reuse=reuse, scope='g_conv_time')
    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm, reuse=reuse, scope='g_conv_spatial')

    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])
    fc = slim.flatten(net)

    net1 = slim.fully_connected(fc, 64, weights_regularizer=None, reuse=reuse, scope='g_fc')
    net2 = slim.fully_connected(net1, 16, reuse=reuse, scope='g_fc1',activation_fn=lrelu) 
    # net4 = slim.fully_connected(net3, 32, reuse=reuse, scope='fc2', activation_fn=tf.nn.leaky_relu)
    net = slim.fully_connected(net1, cls_num, activation_fn=None, reuse=reuse, scope='g_output')
    pred = tf.nn.softmax(net)
    # net4 = slim.fully_connected(net1, 8, reuse=reuse, scope='fea')

    return net, pred, net1, net2







def signal_multitask(input, channel_size, cls_num, is_training):

    weights_regularizer = slim.l2_regularizer(0.01)

    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None)
    # net = slim.conv2d(net, 16, [1, 3], padding='SAME')
    # net = slim.max_pool2d(net, [1, 10],stride=[1,10])

    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm)
    # net = slim.conv2d(net, 32, [1, 3], padding='SAME')
    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])

    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])

    fc = slim.flatten(net)

    fc = slim.fully_connected(fc, 128)
    fc = slim.fully_connected(fc, 64)


    net1 = slim.fully_connected(fc, 64)
    net1 = slim.fully_connected(net1, 32)
    net1 = slim.fully_connected(net1, 9, activation_fn=None)


    net = slim.fully_connected(fc, 64, weights_regularizer=None)
    net = slim.fully_connected(net, 32)
    net = slim.fully_connected(net, cls_num, activation_fn=None)


    pred = tf.nn.softmax(net)
    person = tf.nn.softmax(net1)



    return net, pred, net1, person


def signal_multitask_siamese(input, channel_size, cls_num, is_training, reuse=False):

    weights_regularizer = slim.l2_regularizer(0.01)

    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None, reuse=reuse, scope='conv_time')
    # net = slim.conv2d(net, 16, [1, 3], padding='SAME')
    # net = slim.max_pool2d(net, [1, 10],stride=[1,10])

    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm, reuse=reuse, scope='conv_spatial')
    # net = slim.conv2d(net, 32, [1, 3], padding='SAME')
    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])

    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])

    fc = slim.flatten(net)

    fc = slim.fully_connected(fc, 128, reuse=reuse, scope='fc1')
    fc = slim.fully_connected(fc, 64, reuse=reuse, scope='fc2')


    net1 = slim.fully_connected(fc, 64, reuse=reuse, scope='fc3')
    net1 = slim.fully_connected(net1, 32, reuse=reuse, scope='fc4')
    net1 = slim.fully_connected(net1, 9, activation_fn=None, reuse=reuse, scope='fc5')


    net = slim.fully_connected(fc, 64, weights_regularizer=None, reuse=reuse, scope='fc6')
    net_2 = slim.fully_connected(net, 32, reuse=reuse, scope='fc7')
    net = slim.fully_connected(net_2, cls_num, activation_fn=None, reuse=reuse, scope='fc8')


    pred = tf.nn.softmax(net)
    person = tf.nn.softmax(net1)



    return net, pred, net1, person, net_2


def signal_multitask_rerun(input, channel_size, cls_num, is_training):

    weights_regularizer = slim.l2_regularizer(0.01)

    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None)
    # net = slim.conv2d(net, 16, [1, 3], padding='SAME')
    # net = slim.max_pool2d(net, [1, 10],stride=[1,10])

    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm)
    # net = slim.conv2d(net, 32, [1, 3], padding='SAME')
    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])

    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])

    fc = slim.flatten(net)


    net1 = slim.fully_connected(fc, 32)
    net1 = slim.fully_connected(net1, 9, activation_fn=None)
    person = tf.nn.softmax(net1)

    net = slim.fully_connected(fc, 64, weights_regularizer=None)
    net = slim.fully_connected(net, 32)
    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)



    return net, pred, net1, person






def signal_multitask_fusion(input, channel_size, cls_num, is_training):

    weights_regularizer = slim.l2_regularizer(0.01)

    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None)
    # net = slim.conv2d(net, 16, [1, 3], padding='SAME')
    # net = slim.max_pool2d(net, [1, 10],stride=[1,10])

    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm)
    # net = slim.conv2d(net, 32, [1, 3], padding='SAME')
    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])

    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])

    fc = slim.flatten(net)


    net1_1 = slim.fully_connected(fc, 32)
    net1 = slim.fully_connected(net1_1, 9, activation_fn=None)



    net = slim.fully_connected(fc, 64, weights_regularizer=None)

    net2 = tf.concat([net, net1_1], axis=1)

    net = slim.fully_connected(net2, 32)
    net = slim.fully_connected(net, cls_num, activation_fn=None)

    pred = tf.nn.softmax(net)
    person = tf.nn.softmax(net1)



    return net, pred, net1, person

def signal_more(input, channel_size, cls_num, is_training):

    weights_regularizer = slim.l2_regularizer(0.01)

    net_1 = slim.conv2d(input, 10, [1, 25], stride=[1, 1], padding='SAME', weights_regularizer=None)
    net_2 = slim.conv2d(input, 10, [1, 15], stride=[1, 1], padding='SAME')
    net_3 = slim.conv2d(input, 10, [1, 35], stride=[1, 1], padding='SAME')
    # net = slim.max_pool2d(net, [1, 10],stride=[1,10])
    net = tf.concat([net_1, net_2, net_3], axis=3)

    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID', weights_regularizer=None, normalizer_fn=slim.batch_norm)
    # net = slim.conv2d(net, 30, [1, 3], padding='VALID')
    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])

    # net = slim.conv2d(net, 30, [1, 5], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])

    fc = slim.flatten(net)

    net = slim.fully_connected(fc, 64, weights_regularizer=None)
    net = slim.fully_connected(net, 32)
    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)

    return net, pred


def signal_dense(input, channel_size, cls_num, is_training):

    net = slim.conv2d(input, 30, [1, 25], padding='VALID')

    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID', normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training})

    net = slim.max_pool2d(net, [1, 3], stride=[1, 3])

    x = net

    net = slim.conv2d(net, 40, [1, 1], activation_fn=tf.nn.elu)

    net = slim.dropout(net, is_training=is_training)

    net = tf.concat([x, net], axis=-1)

    x = net

    net = slim.conv2d(net, 40,  [1, 1], activation_fn=tf.nn.elu)

    net = slim.dropout(net, is_training=is_training)

    net = tf.concat([x,net], axis=-1)

    net = slim.max_pool2d(net, [1, 3], [1, 3])

    net = slim.flatten(net)

    net = slim.fully_connected(net, 64)

    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)

    return net, pred





def discriminator(fea, reuse=False):

    fc1 = slim.fully_connected(fea, 64, activation_fn=lrelu, reuse=reuse, scope='d_fc1')
    fc2 = slim.fully_connected(fc1, 32, activation_fn=lrelu, reuse=reuse, scope='d_fc2')
    fc3 = slim.fully_connected(fc2, 16, activation_fn=lrelu, reuse=reuse, scope='d_fc3')
    d_out_logits = slim.fully_connected(fc3, 1, activation_fn=None, reuse=reuse, scope='d_out')

    d_out = tf.nn.sigmoid(d_out_logits)

    return d_out, d_out_logits

