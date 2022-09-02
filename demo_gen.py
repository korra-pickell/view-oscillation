import os, cv2
import numpy as np
import tensorflow as tf
from progress.bar import Bar

data = r'E:\DATA\View Oscillation'

target_size = (512,512)
num_examples = 5

def get_img(path):
    img = cv2.resize(cv2.imread(path),target_size)
    return np.array(img)/255


def get_demo_imgs():
    demo_imgs = np.zeros((num_examples,target_size[0],target_size[1],3))
    for d in range(0,46):
        dpaths = os.listdir(os.path.join(data,str(d)))
        for p in dpaths[:num_examples]:
            d