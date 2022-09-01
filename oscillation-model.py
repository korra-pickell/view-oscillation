import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Reshape, Dropout, Conv2DTranspose, BatchNormalization, LeakyReLU, ReLU, Flatten, Concatenate
from keras.activations import relu, leaky_relu


TARGET_SHAPE = (128,128,3,)
SHUFFLE_BUFFER = 500
BATCH_SIZE = 2
EPOCHS = 3

data_path = r'E:\DATA\View Oscillation 2\128.npz'


def generator(path):
    with np.load(path) as data:
        x0,x1,y0 = data['x0'],data['x1'],data['y0']
        x0 = np.expand_dims(x0,axis=0)
        #x1 = np.expand_dims(x1,axis=0)
        y0 = np.expand_dims(y0,axis=0)
        #print(len(x0),len(x1),len(y0))
        #s=input('...')
        for x_0,x_1,y_0 in zip(x0,x1,y0):
            #print(x_0.shape,x_1.shape,y_0.shape)
            #print(x_1)
            #s=input('...')
            yield {'input_1':x_0, 'input_2':x_1}, y_0


def get_dataset():
    gen = generator(data_path)
    dataset = tf.data.Dataset.from_generator(gen)
    dataset.shuffle(SHUFFLE_BUFFER)

    train_dataset = dataset.take(4200)
    test_dataset = dataset.skip(4200)

    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    return train_dataset,test_dataset



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


def get_branch0():
    input0 = Input(shape=(128,128,3,))

    down_stack = [downsample(32,4,apply_batchnorm=False),
                    downsample(32,4),
                    downsample(32,4),
                    downsample(32,4),
                    downsample(32,4),
                    downsample(32,4)]
                    #downsample(512,4),
                    #downsample(512,4)]

    x = input0

    for down in down_stack:
        x = down(x)
    
    x = Flatten()(x)
    x = Dense(2048,activation='sigmoid')(x)

    model = tf.keras.models.Model(input0,x)
    return model

def get_branch1():
    input1 = Input(shape=(None,1))
    x = Dense(1,activation='sigmoid')(input1)
    model = tf.keras.models.Model(input1,x)
    return model


def get_model_full():

    branch0 = get_branch0()
    branch1 = get_branch1()
    combined_branches = Concatenate()([branch0.output,branch1.output])
    x = Dense(2049,'relu')(combined_branches)
    x = Dense(2048,'sigmoid')(x)
    x = Reshape((2,2,512,))(x)
    
    
    up_stack = [upsample(32,4, apply_dropout=True),
                    upsample(32,4, apply_dropout=True),
                    upsample(32,4, apply_dropout=True),
                    upsample(32,4),
                    upsample(32,4)]
                    #upsample(512,4),
                    #upsample(512,4)]
    
    
    for up in up_stack:
        x = up(x)

    initializer = tf.random_normal_initializer(0., 0.02)

    x = Conv2DTranspose(3,
                        4,
                        strides=2,
                        padding='same',
                        kernel_initializer=initializer,
                        activation='tanh')(x)

    

    return tf.keras.Model(inputs=[branch0.input,branch1.input],outputs=x)


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
    #train_dataset,test_dataset = get_dataset()
    #model.fit(train_dataset, validation_data = test_dataset, epochs=EPOCHS)
    model.fit(generator(data_path),epochs=4)

