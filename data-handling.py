import cv2, os
import numpy as np
from progress.bar import Bar

data = r'E:\DATA\View-Oscillation-5k'
data2 = r'E:\DATA\View Oscillation 2\NPZ-F'
target_size = (256,256)

num_unique_scenes = 5000

def get_img(path):
    img = cv2.resize(cv2.imread(path),target_size)
    return (np.array(img)/127.5)-1

def to_npz():
    bar = Bar(' PROCESSING ', max = num_unique_scenes*46, suffix='%(percent)d%%')
    for d in range(23,24):
        os.mkdir(os.path.join(data2,str(d)))
        for n in range(6,num_unique_scenes):
            x = get_img(os.path.join(data,str(0),str(n).zfill(4)+'.jpg'))
            y = get_img(os.path.join(data,str(d),str(n).zfill(4)+'.jpg'))
            np.savez(os.path.join(data2,str(d),str(n).zfill(5)+'.npz'),
                x = np.array(x),
                y = np.array(y))
        bar.next()
    bar.finish()

to_npz()