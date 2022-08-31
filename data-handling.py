import cv2, os
import numpy as np

data = r'E:\DATA\View Oscillation'
target_size = (512,512)

def get_img(path):
    img = cv2.resize(cv2.imread(path),target_size)
    return np.array(img)/255


def to_npz():
    imgs = []
    for f in range(0,46):
        d_imgs = []
        for im in os.listdir(os.path.join(data,str(f))):
            print(im)
            d_img = get_img(os.path.join(data,im))
            print(d_img)
            s = input('..')




to_npz()