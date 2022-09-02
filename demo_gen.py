import os, cv2
import numpy as np
import tensorflow as tf
from progress.bar import Bar

data = r'E:\DATA\View Oscillation'
models = r'E:\DATA\View Oscillation 2\models'
img_save_dir = r'E:\DATA\View Oscillation 2\Demo'

target_size = (256,256)
num_examples = 5

def get_img(path):
    img = cv2.resize(cv2.imread(path),target_size)
    return (np.array(img)/127.5)-1


def get_demo_imgs():
    demo_imgs = np.zeros((num_examples,46,target_size[0],target_size[1],3))
    ex_num = 1
    for ex in range(num_examples):
        for d in range(0,46):
            demo_imgs[ex,d] = get_img(os.path.join(data,str(d),str(ex_num).zfill(4)+'.jpg'))
    return demo_imgs


def get_model():
    model_name = os.listdir(models)[0]
    model_path = os.path.join(models,model_name)
    #print('----  LOADING MODEL :  {}  ----'.format(model_name))
    return tf.keras.models.load_model(model_path)


def save_demo(img, index, pred=True):
    img = img.numpy()
    img = ((img+1)*127.5).astype(int)
    #print(img)
    #s=input('..')
    if pred==True:
        cv2.imwrite(os.path.join(img_save_dir,'pred',str(index)+'.jpg'),img)
    else:
        cv2.imwrite(os.path.join(img_save_dir,'true',str(index)+'.jpg'),img)


if __name__ == '__main__':
    model = get_model()
    demo_imgs = get_demo_imgs()

    img_pred0 = model([[np.expand_dims(demo_imgs[0][0],axis=0),np.expand_dims(np.array(0.0),axis=0)]])[0]
    
    save_demo(img_pred0,0,pred=True)