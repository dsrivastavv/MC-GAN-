from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from keras.callbacks import TensorBoard
import cv2
import tensorflow as tf
import pdb

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def get_condition_image(imgs):
    batch_size = imgs.shape[0]
    cond_imgs = []
    for i in range(batch_size):
        img = imgs[i]
        num_cond = np.random.randint(4, 8, 1)
        cond_idx = np.random.randint(0, 26, num_cond)
        cond_img = np.ones((64,64,78))
        for j in cond_idx:
            cond_img[:,:,j*3:j*3+3] = img[:,:,j*3:j*3+3]
        cond_imgs.append(cond_img)
    cond_imgs = np.array(cond_imgs)
    return cond_imgs

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 78
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 1
        self.df = 1

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        grd_img = Input(shape=self.img_shape)
        cond_img = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(cond_img)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, cond_img])

        self.combined = Model(inputs=[grd_img, cond_img], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)


        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*8)
        u2 = deconv2d(u1, d2, self.gf*4)
        u3 = deconv2d(u2, d1, self.gf*2)
        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        Model(d0, output_img).summary()
        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        grd_img = Input(shape=self.img_shape)
        cond_img = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([grd_img, cond_img])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        Model([grd_img, cond_img], validity).summary()
        return Model([grd_img, cond_img], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        data_dir = 'datasets/Capitals_colorGrad64/train/'
        files = os.listdir(data_dir)
        gt_imgs = []
        for count, file in enumerate(files):
            if count >=50:
                break
            img = cv2.imread(data_dir + file)
            new_img = []
            for i in range(26):
                new_img.append(img[:,64*i:64*(i+1),0])
                new_img.append(img[:,64*i:64*(i+1),1])
                new_img.append(img[:,64*i:64*(i+1),2])
            new_img = np.array(new_img)
            new_img = np.transpose(new_img , (1,2,0))
            gt_imgs.append(new_img)

        gt_imgs = np.array(gt_imgs)
        gt_imgs = ( gt_imgs.astype( np.float32 ) - 127.5 ) / 127.5

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        log_path = 'graph/imageTranslation'
        callback = TensorBoard(log_path)
        callback.set_model(self.combined)
        train_names = ['dloss' , 'dacc' ,'gloss']


        for epoch in range(epochs):

            idx = np.random.randint(0, gt_imgs.shape[0], batch_size)
            gt_imgs_batch = gt_imgs[idx]
            cond_imgs_batch = get_condition_image(gt_imgs_batch)
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Condition on B and generate a translated version
            fake_A = self.generator.predict(cond_imgs_batch)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch([gt_imgs_batch, cond_imgs_batch], valid)
            d_loss_fake = self.discriminator.train_on_batch([fake_A, cond_imgs_batch], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            #  Train Generator
            # -----------------

            # Train the generators
            g_loss = self.combined.train_on_batch([gt_imgs_batch, cond_imgs_batch], [valid, gt_imgs_batch])

            elapsed_time = datetime.datetime.now() - start_time

            write_log(callback, train_names, np.asarray([d_loss[0] , 100*d_loss[1], g_loss[0]]), epoch)
            # Plot the progress
            print ("[Epoch %d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch,d_loss[0], 100*d_loss[1],g_loss[0],str(elapsed_time)))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch,batch_size, cond_imgs_batch, gt_imgs_batch)
            if epoch % sample_interval*10 == 0:
                self.save_model(epoch)                

    def sample_images(self, epoch , batch_size , cond_imgs, gt_imgs):
        direc = 'results/imageTranslation/'
        os.makedirs(direc , exist_ok=True)
        idx = np.random.randint(0, cond_imgs.shape[0], batch_size)
        cond_imgs_batch = cond_imgs[idx]
        gt_imgs_batch = gt_imgs[idx]

        fake_imgs_batch = self.generator.predict(cond_imgs_batch)


        for bs in range(cond_imgs.shape[0]):
            r, c = 3, int(cond_imgs.shape[3]/3)
            cond_img = cond_imgs_batch[bs]
            gt_img = gt_imgs_batch[bs]
            fake_img = fake_imgs_batch[bs]

            # Rescale images 0 - 1
            fake_img = 0.5*fake_img + 0.5
            cond_img = 0.5*cond_img + 0.5
            gt_img = 0.5*gt_img + 0.5

            titles = ['Condition', 'Generated', 'Original']
            fig, axs = plt.subplots(r, c)
            for i in range(r):
                for j in range(c):
                    if i==0:
                        axs[i,j].imshow(cond_img[:,:,3*j:3*j+3])
                    if i==1:
                        axs[i,j].imshow(fake_img[:,:,3*j:3*j+3])

                    if i==2:
                        axs[i,j].imshow(gt_img[:,:,3*j:3*j+3])
                        
                    # axs[i, j].set_title(titles[i])
                    axs[i,j].axis('off')
            fig.savefig(direc + "/%s/%d_%d.png" % (self.dataset_name, epoch,bs))
            plt.close()

    def save_model(self, epoch):

        def save(model, model_name):
            save_dir = "saved_model/imageTranslation/"
            os.makedirs(save_dir, exist_ok=True)
            model_path = save_dir + "/%s.json" % model_name
            weights_path = save_dir + "/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator" + str(epoch))
        save(self.discriminator, "discriminator"+ str(epoch))            


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=200000, batch_size=10, sample_interval=200)
    gan.test()
