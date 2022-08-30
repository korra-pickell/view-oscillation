
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU
from keras.activations import relu, leaky_relu

TARGET_SIZE = (512,512,3)

def downsample(filters, size, apply_batchnorm=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = Conv2D(filters,size,strides=2,padding='same',kernel_initializer=initializer, use_bias=False)
    if apply_batchnorm:
        x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x


def get_model_branch0():
    input0 = Input(shape=(512,512,3,))


def get_model_branch1():
    pass

def get_model_full():
    pass