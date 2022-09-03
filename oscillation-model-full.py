import os, time, cv2, datetime

import numpy as np
import tensorflow as tf

from progress.bar import Bar
from PIL import Image
from matplotlib import pyplot as plt
from IPython import display

wkdir = os.path.dirname(os.path.realpath(__file__))

data = r'E:\DATA\View Oscillation'
data_path = r'E:\DATA\View Oscillation 2\NPZ-F'
img_save_dir = r'E:\DATA\View Oscillation 2\Demo'

checkpoint_dir = ''

NUM_EPOCHS = 5
MAX_SAMPLES = 100
num_demo_examples = 1
GEN_FILTER_SIZE = 4

BUFFER_SIZE = 3000
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 512

OUTPUT_CHANNELS = 3

TARGET_SHAPE = (IMG_WIDTH,IMG_HEIGHT,OUTPUT_CHANNELS,)



def get_img(path):
    img = cv2.resize(cv2.imread(path),(TARGET_SHAPE[0],TARGET_SHAPE[1]))
    return (np.array(img)/127.5)-1


def get_demo_imgs():
    demo_imgs = np.zeros((num_demo_examples,46,TARGET_SHAPE[0],TARGET_SHAPE[1],OUTPUT_CHANNELS))
    ex_num = 1
    for ex in range(num_demo_examples):
        for d in range(0,46):
            demo_imgs[ex,d] = get_img(os.path.join(data,str(d),str(ex_num).zfill(4)+'.jpg'))
    return demo_imgs


def save_demo(img, index, pred=True):
    img = img.numpy()[0]
    img = ((img+1)*127.5).astype(int)
    #print(img.shape)
    #s=input('..')
    if pred==True:
        cv2.imwrite(os.path.join(img_save_dir,'pred',str(index)+'.jpg'),img)
    else:
        cv2.imwrite(os.path.join(img_save_dir,'true',str(index)+'.jpg'),img)


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                                         kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                                                        padding='same',
                                                                        kernel_initializer=initializer,
                                                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS])

    down_stack = [
        downsample(64, GEN_FILTER_SIZE, apply_batchnorm=False),    # (bs, 128, 128, 64)
        downsample(128, GEN_FILTER_SIZE),    # (bs, 64, 64, 128)
        downsample(256, GEN_FILTER_SIZE),    # (bs, 32, 32, 256)
        downsample(512, GEN_FILTER_SIZE),    # (bs, 16, 16, 512)
        downsample(512, GEN_FILTER_SIZE),    # (bs, 8, 8, 512)
        downsample(512, GEN_FILTER_SIZE),
        downsample(512, GEN_FILTER_SIZE),    # (bs, 4, 4, 512)
        downsample(512, GEN_FILTER_SIZE),    # (bs, 2, 2, 512)
        downsample(512, GEN_FILTER_SIZE),    # (bs, 1, 1, 512)
        #downsample(512, 4),    # (bs, 1, 1, 512)
    ]

    up_stack = [
        #upsample(512, 4, apply_dropout=True),    # (bs, 2, 2, 1024)
        upsample(512, GEN_FILTER_SIZE, apply_dropout=True),    # (bs, 2, 2, 1024)
        upsample(512, GEN_FILTER_SIZE, apply_dropout=True),    # (bs, 4, 4, 1024)
        upsample(512, GEN_FILTER_SIZE, apply_dropout=True),    # (bs, 8, 8, 1024)
        upsample(512, GEN_FILTER_SIZE),
        upsample(512, GEN_FILTER_SIZE),    # (bs, 16, 16, 1024)
        upsample(256, GEN_FILTER_SIZE),    # (bs, 32, 32, 512)
        upsample(128, GEN_FILTER_SIZE),    # (bs, 64, 64, 256)
        upsample(64, GEN_FILTER_SIZE),    # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, GEN_FILTER_SIZE,
                                                                                 strides=2,
                                                                                 padding='same',
                                                                                 kernel_initializer=initializer,
                                                                                 activation='tanh')    # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()

tf.keras.utils.plot_model(
    generator,
    to_file=r'E:\Documents\PRGM\NEURAL\View Oscillation\model_graphics\gen_model.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',
    dpi=64
)

LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(target - tf.cast(gen_output,dtype='float64')))

    total_gen_loss = tf.cast(gan_loss,dtype='float64') + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])    # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)    # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)    # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)    # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)    # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,kernel_initializer=initializer,use_bias=False)(zero_pad1)    # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)    # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,kernel_initializer=initializer)(zero_pad2)    # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()

tf.keras.utils.plot_model(
    discriminator,
    to_file=r'E:\Documents\PRGM\NEURAL\View Oscillation\model_graphics\dis_model.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',
    dpi=64
)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


def generate_images(model, test_input, tar, epoch):

    predictions = []
    gen_bar = Bar('Generating ',max = len(test_input))
    for t in test_input:
        gen_bar.next()
        pred = model(t, training=True)
        predictions.append(pred)
    gen_bar.finish()
    fig=plt.figure(figsize=(15, 15),dpi=150)
    fig.tight_layout()

    display_list = [[test_input[0], predictions[0], tar[0]],
                    [test_input[1], predictions[1], tar[1]],
                    [test_input[2], predictions[2], tar[2]]]
    columns,rows = 3,3
    title = ['Input Image', 'Predicted Image', 'Ground Truth']

    for i in range(columns*rows):
        img = (((np.array(display_list[int((i/columns)%rows)][int(i%columns)][0])*0.5)+0.5)*255).astype('uint8')
        fig.add_subplot(rows,columns,i+1)
        plt.title(title[int(i%columns)])
        plt.imshow(img)
        plt.axis('off')
    plt.savefig('E:\\DATA\\img_output\\3d-object-removal\\output_'+str(epoch)+'.png')
    plt.close()



#log_dir="logs/"

#summary_writer = tf.summary.create_file_writer(
#log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, input_condition, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))


def fit(train_ds, epochs, test_ds=None):
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        #for example_input, example_target in test_ds.take(1):
        #generate_images(generator, test_dataset_x, test_dataset_y, file_index, epoch_number)

        # Train
        train_bar = Bar('Training ',max = len(train_ds), suffix = '%(percent).1f%% - %(eta)ds')
        for n, (inputs, target) in enumerate(train_ds):
            train_bar.next()
            #print(input_image.shape,target.shape)
            #stop = input('.......')
            train_step(inputs['input_1'], inputs['input_2'], target, epoch)
        train_bar.finish()
        print()

        # saving (checkpoint) the model every 20 epochs
        #if (epoch + 1) % 20 == 0:
            #checkpoint.save(file_prefix=checkpoint_prefix)
        demo_images = get_demo_imgs()
        demo_output = generator(tf.expand_dims(demo_images[0][0],axis=0),training=True)
        save_demo(demo_output,epoch,pred=True)
        #s = input('...')
        #generate_images(generator, test_dataset_x, test_dataset_y, epoch_number)

        #print('SAVING CKPT')
        #checkpoint.save(file_prefix=checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,time.time()-start))


def get_data_ids():
    all_ids = os.listdir(data_path)[:MAX_SAMPLES]
    val_point = int(len(all_ids)*0.8)
    return all_ids[:val_point], all_ids[val_point:]


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, dpath ,batch_size=32, dim1=(TARGET_SHAPE[0],TARGET_SHAPE[1]), dim2=(1,), n_channels=3, shuffle=True):
        'Initialization'
        self.dpath = dpath
        self.dim1 = dim1
        self.dim2 = dim2
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X0 = np.empty((self.batch_size, *self.dim1, self.n_channels))
        X1 = np.empty((self.batch_size, *self.dim2))
        y = np.empty((self.batch_size, *self.dim1, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data = np.load(os.path.join(self.dpath,ID))
            x0,x1,y0 = data['x0'],data['x1'],data['y0']
            x0 = cv2.resize(x0,(TARGET_SHAPE[0],TARGET_SHAPE[1]))
            y0 = cv2.resize(x0,(TARGET_SHAPE[0],TARGET_SHAPE[1]))
            X0[i,], X1[i,], y[i,] = x0, x1, y0

        return {'input_1':X0, 'input_2':X1}, y


if __name__ == '__main__':
    train_ids, val_ids = get_data_ids()

    train_gen = DataGenerator(train_ids, data_path,batch_size=BATCH_SIZE)

    fit(train_ds = train_gen, epochs = NUM_EPOCHS)
