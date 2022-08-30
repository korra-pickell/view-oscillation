
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Conv2DTranspose, BatchNormalization, LeakyReLU, ReLU, Flatten, Concatenate
from keras.activations import relu, leaky_relu

TARGET_SIZE = (512,512,3)

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = Conv2D(filters,
                size,
                strides=2,
                padding='same',
                kernel_initializer=initializer,
                use_bias=False)

    if apply_batchnorm:
        x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = Conv2DTranspose(filters,
                        size,
                        strides=2,
                        padding='same',
                        kernel_initializer=initializer,
                        use_bias=False)

    x = BatchNormalization()(x)
    if apply_dropout:
        x = Dropout(0.5)(x)
    x = ReLU()(x)
    return x

def get_model_branch0():
    input0 = Input(shape=(512,512,3,))

    '''512'''
    x = downsample(64,4,apply_batchnorm=False)(input0)
    '''256'''
    x = downsample(64,4)
    '''128'''
    x = downsample(64,4)
    '''64'''
    x = downsample(64,4)
    '''32'''
    x = downsample(64,4)
    '''16'''
    x = downsample(64,4)
    '''8'''
    x = downsample(64,4)
    '''4'''
    x = Flatten()(x)

    return x


def get_model_branch1():
    input1 = Input(shape=(1,))

    return input1

def get_model_full():
    pass