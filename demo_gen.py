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
    demo_imgs = np.zeros((num_examples,46,target_size[0],target_size[1],3))
    ex_num = 1
    for ex in range(num_examples):
        for d in range(0,46):
            demo_imgs[ex,d] = get_img(os.path.join(data,str(d),str(ex_num).zfill(4)+'.jpg'))



get_demo_imgs()