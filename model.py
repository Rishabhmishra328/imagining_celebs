import numpy as np
import time
import matplotlib.pyplot as plt
import data_preprocessing as dp
import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.initializers import glorot_normal
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials
import zipfile
from progress.bar import Bar
from os import listdir
import cv2
import h5py
import os.path
import sys
import time
# from PIL import Image
from IPython.display import Image
import random
from google_drive_downloader import GoogleDriveDownloader as gdd
from scipy.misc import imsave




# 1. Authenticate and create the PyDrive client.
data_zip = 'data.zip'
unzip_dir = 'data/img/'
img_dir = unzip_dir + 'img_align_celeba/'
clean_data_dir = unzip_dir + 'clean_data/'
preview_filename = ''

if not os.path.isfile(data_zip):
	gdd.download_file_from_google_drive(file_id='0B7EVK8r0v71pZjFTYXZWM3FlRnM', dest_path='./data.zip', unzip=False)
else:
	print 'Dataset exists, Processing ...'

zip_ref = zipfile.ZipFile(data_zip, 'r')

if not os.path.exists(unzip_dir):
  os.makedirs(unzip_dir)
  print 'Extracting ...'
  zip_ref.extractall(unzip_dir)
  print 'Finished ...'
else:
  print 'Already extracted please check again, delete data folder if error occurs'
  print 'Processing ...'
  
zip_ref.close()

# 2. Data preperation
def make_file(full = False, batch_size = 1000):
    if not os.path.exists(clean_data_dir):
        os.makedirs(clean_data_dir)

        image_files = listdir(img_dir)
	if full:
		batch_size = len(image_files)

	data_shape = (218,178,3) #image dimensions      

        im_file_pbar = Bar('File Progress', max = len(image_files))

        for iterations in range((len(image_files) / batch_size)-1):
        	h5_filepath = clean_data_dir + 'data_'+str(iterations)+'.h5'
        	print h5_filepath
        	h5f = h5py.File(h5_filepath, 'w')
        	train_data = np.empty((batch_size, 218, 178, 3)) #image dimensions
        	for index in range(len(image_files[iterations:iterations + batch_size])):
        	    preview_filename = img_dir + image_files[index]
        	    im = cv2.imread(img_dir + image_files[index])
            

                im = np.asarray(im, dtype = np.float32)
            	train_data[index] = im
            
            	im_file_pbar.next()
        
        	train_data = np.asarray(train_data, dtype = np.float32)
        
        	h5f.create_dataset('train', data = train_data)
        	print iterations, train_data.shape
        
        
        	print 'Image preview ...'
        	Image(filename = preview_filename)
            
        	h5f.close()
    else:
	print 'Folder exists. please manullay delete files in /data/img/clean_data if error occurs'
	print 'Processing ...'

make_file(batch_size = 500) #making files

# 3. Data retrieval
def get_data(full = False, batch_size = 1000):
    file_index = random.randint(0,(len(listdir(img_dir))/ batch_size - 2))
    if full:
      file_index = 0
    h5_filepath = clean_data_dir + 'data_'+str(file_index)+'.h5'
    data_file = h5py.File(h5_filepath,'r')
    data = data_file.get('train')
    
    data = np.asarray(data, dtype = np.float32)

#     data = np.swapaxes(data,1,2)
#     data = np.swapaxes(data,2,3)
#     print type(data)
    
#     plt.imshow(data[0])
  
    return data

print get_data(True, batch_size = 500).shape

# 4. Data management
directory = clean_data_dir
dat_files = os.listdir(directory)

# for file in dat_files:
#   if file.endswith(".h5"):
#     os.remove(directory + '/' + file)
#     print 'Removing ...', file

# 5. Finally the model
class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self, steps = 1):
        print("Elapsed: %s " % self.elapsed((time.time() - self.start_time) * steps) )

class DCGAN(object):
    def __init__(self, img_rows=218, img_cols=178, channel=3):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4

        input_shape = (self.img_rows, self.img_cols, self.channel)
        print input_shape
        self.D.add(Conv2D(depth*1, kernel_size = 5, strides=5, input_shape = input_shape, padding='same', data_format = 'channels_last'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, kernel_size = 2, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, kernel_size = 2, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, kernel_size = 4, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        h_dim = 3
        w_dim = 1

        self.G.add(Dense(h_dim*w_dim*depth, input_dim=1000, kernel_initializer = glorot_normal()))
        self.G.add(BatchNormalization(momentum = 0.5))
        self.G.add(Activation('tanh'))
        self.G.add(Reshape((h_dim, w_dim, depth)))
        self.G.add(Dropout(dropout))

        self.G.add(Conv2DTranspose(int(depth/2), kernel_size = (2,2), strides = 2,padding='valid', name = 'gen_conv_1', kernel_initializer = glorot_normal()))
        self.G.add(BatchNormalization(momentum = 0.5))
        self.G.add(Activation('tanh'))

        self.G.add(Conv2DTranspose(int(depth/4), kernel_size = (3,3), strides = 2,padding='valid', name = 'gen_conv_2', kernel_initializer = glorot_normal()))
        self.G.add(BatchNormalization(momentum = 0.5))
        self.G.add(Activation('tanh'))
  
        self.G.add(Conv2DTranspose(int(depth/4), kernel_size = (2,2), strides = 2,padding='valid', name = 'gen_conv_3', kernel_initializer = glorot_normal()))
        self.G.add(BatchNormalization(momentum = 0.5))
        self.G.add(Activation('tanh'))

        self.G.add(Conv2DTranspose(int(depth/8), kernel_size = (4,8), strides = (2,4),padding='valid', name = 'gen_conv_4', kernel_initializer = glorot_normal()))
        self.G.add(BatchNormalization(momentum = 0.5))
        self.G.add(Activation('tanh'))

        self.G.add(Conv2DTranspose(3, kernel_size = (3,3), strides = 2 ,padding='valid', name = 'gen_conv_5', kernel_initializer = glorot_normal()))
        self.G.add(BatchNormalization(momentum = 0.5))
        self.G.add(Activation('tanh'))
        
        self.G.add(Conv2DTranspose(3, kernel_size = 2, strides = 2 ,padding='same', name = 'gen_conv_6', kernel_initializer = glorot_normal()))
        self.G.add(BatchNormalization(momentum = 0.5))
        self.G.add(Activation('relu'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = Adam(lr=5e-7, decay = 5e-9)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = Adam(lr=5e-7, decay = 5e-9)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,
            metrics=['accuracy'])
        return self.AM

class CELEB_DCGAN(object):
    def __init__(self):
        self.img_rows = 218
        self.img_cols = 178
        self.channel = 3

        # print self.x_train

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=1000, save_interval=0, noise_size = 1000):
        noise_input = None
        print 'Training ...'
        d_loss_line, a_loss_line = [], []
        for i in range(train_steps):
            timer = ElapsedTimer()

            self.x_train = get_data(batch_size = batch_size)
            self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, self.channel).astype(np.float32)

            self.discriminator.trainable = True
            
            images_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(0.0, 1.0, size=[batch_size, noise_size])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)
            d_loss_line.append(d_loss[0])
            
            self.discriminator.trainable = False
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(0.0, 1.0, size=[batch_size, noise_size])
            a_loss = self.adversarial.train_on_batch(noise, y)
            a_loss_line.append(a_loss[0])
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])

            if (save_interval>0 and i>0):
                if (i)%( 1 * save_interval)==0:
                  noise_input =  np.random.uniform(0.0, 1.0, size=[batch_size, noise_size])

                  self.plot_images(save2file=True, samples=2, noise=noise_input, step=(i+1))
                  self.plot_images(save2file=False, samples=2, step=(i+1), fake = False)
                  loss_line_indices = np.arange(len(d_loss_line))
                  plt.figure(1)
                  plt.subplot(211)
                  plt.plot(loss_line_indices, d_loss_line)
                  plt.title('D loss')
                  plt.subplot(212)
                  plt.plot(loss_line_indices, a_loss_line)
                  plt.title('A loss')
		  filename = "./res/loss_plot_%d.png" % i
	          plt.savefig(filename)
#                  plt.legend(['D loss', 'A loss'], loc='upper left')
#                  plt.show()
#                  printm()
            
            if (save_interval>0 and i>0):
                if (i)%save_interval==0:
                    print(log_mesg)                    
                    timer.elapsed_time(steps = save_interval)


    def plot_images(self, save2file=False, fake=True, samples=1, noise=None, step=5, noise_size = 1000):
        if not os.path.exists('./res'):
            os.makedirs('./res')
        filename = './res/celeba_gans.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, noise_size])
            else:
                filename = "./res/celeba_gans_%d.png" % step
            images = self.generator.predict(noise)

        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]
	    filename = "./res/celeba_gans_%d_real.png" % step

#        i = np.random.randint(0, images.shape[0], samples)
        image = images[1, :, :, :]
#        image = np.reshape(image, [self.img_rows, self.img_cols, self.channel])

        if save2file:
            print 'Saved at ...', filename
	    imsave(filename, image)

if __name__ == '__main__':
    celeb_gan = CELEB_DCGAN()
    timer = ElapsedTimer()
    celeb_gan.train(train_steps=50000, batch_size=500, save_interval=1)
    timer.elapsed_time()
    celeb_gan.plot_images(fake=True)
    celeb_gan.plot_images(fake=False, save2file=True)
