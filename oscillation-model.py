import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Reshape, Dropout, Conv2DTranspose, BatchNormalization, LeakyReLU, ReLU, Flatten, Concatenate
from keras.activations import relu, leaky_relu

TARGET_SHAPE = (128,128,3,)
SHUFFLE_BUFFER = 500
BATCH_SIZE = 128

data_path = r'E:\DATA\View Oscillation 2\128.npz'


def generator(path):
    with np.load(path) as data:
        x0,x1,y0 = data['x0'],data['x1'],data['y0']
        for x_0,x_1,y_0 in zip(x0,x1,y0):
            yield {"input_1":x_0, "input_2": x_1}, y_0


def get_dataset():
    gen = generator(data_path)
    dataset = tf.data.Dataset.from_generator(gen)
    dataset.shuffle(SHUFFLE_BUFFER)

    train_dataset = dataset.take(4200)
    test_dataset = dataset.skip(4200)

    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    return train_dataset,test_dataset


def load_data():
    with np.load(data_path) as data:
        x0,x1,y0 = data['x0'],data['x1'],data['y0']
    full_dataset = tf.data.Dataset.from_tensor_slices(([x0,x1], y0))

    full_dataset = full_dataset.shuffle(1000)
    
    train_dataset = full_dataset.take(4200)
    test_dataset = full_dataset.skip(4200)

    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    
    return train_dataset, test_dataset


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
            Conv2D(filters, size, strides=2, padding='same',
                                                         kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        Conv2DTranspose(filters, size, strides=2,
                                                                        padding='same',
                                                                        kernel_initializer=initializer,
                                                                        use_bias=False))

    result.add(BatchNormalization())

    if apply_dropout:
            result.add(Dropout(0.5))

    result.add(ReLU())

    return result


def get_model_full():
    input0 = Input(shape=TARGET_SHAPE)
    input1 = Input(shape=(1,))

    down_stack = [downsample(64,4,apply_batchnorm=False),
                    downsample(128,4),
                    downsample(256,4),
                    downsample(512,4),
                    downsample(512,4),
                    downsample(512,4)]
                    #downsample(512,4),
                    #downsample(512,4)]

    up_stack = [upsample(64,4, apply_dropout=True),
                    upsample(128,4, apply_dropout=True),
                    upsample(256,4, apply_dropout=True),
                    upsample(512,4),
                    upsample(512,4)]
                    #upsample(512,4),
                    #upsample(512,4)]

    x = input0
    
    #skips = []

    for down in down_stack:
        x = down(x)
        #skips.append(x)

    #x = Concatenate()([x,input1])

    x = Flatten()(x)
    x = Concatenate()([x,input1])
    x = Dense(2048)(x)
    x = Reshape(target_shape=(2,2,512))(x)

    #skips = reversed(skips[:-1])

    for up in up_stack:
        x = up(x)
        #x = Concatenate()([x,skip])

    #for up, skip in zip(up_stack,skips):
        #x = up(x)
        #x = Concatenate()([x,skip])

    initializer = tf.random_normal_initializer(0., 0.02)

    x = Conv2DTranspose(3,
                        4,
                        strides=2,
                        padding='same',
                        kernel_initializer=initializer,
                        activation='tanh')(x)

    

    return tf.keras.Model(inputs=[input0,input1],outputs=x)


def save_graph(model):
    spath = r'E:\Documents\PRGM\NEURAL\View Oscillation\model.png'
    tf.keras.utils.plot_model(model,
                            to_file=spath,
                            show_shapes=True,
                            show_layer_names=True,
                            rankdir='TB',
                            dpi=64)
    print('Graph Saved')



if __name__ == '__main__':
    
    model = get_model_full()
    save_graph(model)
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mean_squared_error'])
    train_dataset,test_dataset = load_data()
    model.fit(generator(data_path),epochs=4)

