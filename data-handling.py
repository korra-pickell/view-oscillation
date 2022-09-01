import cv2, os
import numpy as np
from progress.bar import Bar

data = r'E:\DATA\View Oscillation'
data2 = r'E:\DATA\View Oscillation 2'
target_size = (128,128)

num_unique_scenes = 100

def get_img(path):
    img = cv2.resize(cv2.imread(path),target_size)
    return np.array(img)/255


def to_npz():
    bar = Bar(' PROCESSING ', max = num_unique_scenes, suffix='%(percent)d%%')
    example_index = 0
    for n in range(1,num_unique_scenes):
        d_imgs = []
        for d in range(0,46):
            d_imgs.append(get_img(os.path.join(data,str(d),str(n).zfill(4)+'.jpg')))
        for i,img in enumerate(d_imgs):
            x0 = d_imgs[0]
            x1 = float(i)/46
            y0 = img
            np.savez(os.path.join(data2,str(example_index).zfill(5)+'.npz'),
                x0 = np.array(x0), 
                x1 = np.array(x1),
                y0 = np.array(y0))
            example_index += 1
        bar.next()
    bar.finish()
        



to_npz()