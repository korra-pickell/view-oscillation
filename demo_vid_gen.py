import cv2, os
import tensorflow as tf
import numpy as np
from oscillation_model_full import downsample, upsample, get_img, save_demo

IMG_WIDTH = 256
IMG_HEIGHT = 256

OUTPUT_CHANNELS = 3

TARGET_SHAPE = (IMG_WIDTH,IMG_HEIGHT,OUTPUT_CHANNELS,)

model_dir = r'E:\DATA\View Oscillation 2\models'
demo_img = r'E:\DATA\View Oscillation 2\demo_final_img.jpg'
demo_dir = r'E:\DATA\View Oscillation 2\Demo\pred'

def render_angle(img,d):

    gen = tf.keras.models.load_model(model_dir)
    img = get_img(img)
    print(img)
    gen_output = gen(tf.expand_dims(img,axis=0),training=True)
    print(gen_output)



if __name__ == '__main__':
    
    angles = [x for x in range(0,1)]
    #angles.remove(10)
    
    input_img = tf.expand_dims(get_img(demo_img),axis=0)

    for d in angles:

        gen = tf.keras.models.load_model(os.path.join(model_dir,str(d)+'.h5'))

        gen_output = gen(input_img,training=True)

        save_demo(gen_output,d,pred=True)