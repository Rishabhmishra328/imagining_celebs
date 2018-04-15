from progress.bar import Bar
from os import listdir
import numpy as np
import cv2
import h5py
import os.path


def get_real_data():
    
    if os.path.isfile('data.npy'):
            with open('data.npy', 'r') as data_file:
                return np.load(data_file)        

    else:
        path = './img_align_celeba/'
            
        image_files = listdir(path)
            
        x = []
        x = np.asarray(x, dtype = np.float32)
            
        im_file_pbar = Bar('File Progress', max = len(image_files))
        
        for index in range(len(image_files)):
            if index == 0:
                im = cv2.imread(path + image_files[index])
                im = np.asarray(im, dtype = np.float32)
                x = im
            else:
                im = cv2.imread(path + image_files[index])
                im = np.asarray(im, dtype = np.float32)
                np.concatenate((x,im), axis = 0)
                im_file_pbar.next()
            
            
        with open('data.npy', 'w') as data_file:
            np.save(data_file, x)
        
        return x

# data = get_real_data()
# h5f = h5py.File('data.h5', 'w')
# h5f.create_dataset('real_data', data = data)
# print 'Saving data.h5 ...'
# h5f.close()