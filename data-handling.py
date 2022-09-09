import cv2, os
import numpy as np
from progress.bar import Bar

data = r'E:\DATA\View Oscillation'
data2 = r'E:\DATA\View Oscillation 2\NPZ-F'
target_size = (128,128)

num_unique_scenes = 100

def get_img(path):
    img = cv2.resize(cv2.imread(path),target_size)
    return (np.array(img)/127.5)-1
'''
def to_npz():
    bar = Bar(' PROCESSING ', max = num_unique_scenes, suffix='%(percent)d%%')
    example_index = 5
    for n in range(example_index+1,num_unique_scenes):
        d_imgs = []
        f_index = 0
        
        for d in range(0,46):
            d_imgs.append(get_img(os.path.join(data,str(d),str(n).zfill(4)+'.jpg')))
        for i,img in enumerate(d_imgs):
            x0 = d_imgs[0]
            y0 = img
            if f_index < 46:
                os.mkdir(os.path.join(data2,str(f_index)))
            f_index+= 1
            
            np.savez(os.path.join(data2,str(i),str(example_index).zfill(5)+'.npz'),
                x0 = np.array(x0),
                y0 = np.array(y0))
            example_index += 1
        #s = input('....')
        bar.next()
    bar.finish()'''


def to_npz():
    bar = Bar(' PROCESSING ', max = num_unique_scenes, suffix='%(percent)d%%')
    example_index = 5
    init_folders = True
    for d in range(0,46):
        if init_folders:
            os.mkdir(os.path.join(data2,str(d)))
        for n in range(example_index+1,num_unique_scenes):
            x = get_img(os.path.join(data,str(0),str(n).zfill(4)+'.jpg'))
            y = get_img(os.path.join(data,str(d),str(n).zfill(4)+'.jpg'))
            np.savez(os.path.join(data2,str(d),str(example_index).zfill(5)+'.npz'),
                x = np.array(x),
                y = np.array(y))
        example_index += 1

to_npz()