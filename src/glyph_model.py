from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate
from keras.layers import BatchNormalization, Activation, Embedding
from keras.layers import ZeroPadding2D, DepthwiseConv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D , Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import keras.backend as K
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pdb
import os
import cv2

def get_condition_image(imgs):
	batch_size = imgs.shape[0]
	cond_imgs = []
	for i in range(batch_size):
		img = imgs[i]
		num_cond = np.random.randint(4, 8, 1)
		cond_idx = np.random.randint(0, 26, num_cond)
		cond_img = np.ones((64,64,26))
		cond_img[:,:,cond_idx] = img[:,:,cond_idx]
		cond_imgs.append(cond_img)

	cond_imgs = np.array(cond_imgs)
	return cond_imgs


def write_log(callback, names, logs, batch_no):
	for name, value in zip(names, logs):
		summary = tf.Summary()
		summary_value = summary.value.add()
		summary_value.simple_value = value
		summary_value.tag = name
		callback.writer.add_summary(summary, batch_no)
		callback.writer.flush()

class GLYPH_MODEL():
	def __init__(self):
		# Input shape
		self.img_rows = 64
		self.img_cols = 64
		self.channels = 26
		self.img_shape = (self.img_rows, self.img_cols, self.channels)

		# # Calculate output shape of D (PatchGAN)
		# Number of filters in the first layer of G and D
		self.gf = 64
		self.df = 52

		optimizer = Adam(0.0002, 0.5)
		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss=['mse','mse'],
			optimizer=optimizer)


		# define output shape of discriminator
		patch_local = 4
		patch_global = 1
		self.disc_patch_local = (patch_local, patch_local, 1)
		self.disc_patch_global = (patch_global, patch_global, 1)


		#-------------------------
		# Construct Computational
		#   Graph of Generator
		#-------------------------

		# Build the generator
		self.generator = self.build_generator()

		# Input images and their conditioning images
		gt_img = Input(shape=self.img_shape)
		cond_img = Input(shape=self.img_shape)

		# By conditioning on B generate a fake version of A
		fake_img = self.generator(cond_img)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# Discriminators determines validity of translated images / condition pairs
		valid_local , valid_global = self.discriminator([fake_img, cond_img])

		self.combined = Model(inputs=[gt_img, cond_img], outputs=[valid_local ,\
												 valid_global, fake_img])
		self.combined.compile(loss=['mse', 'mse', 'mae'],
							  loss_weights=[1, 10, 100],
							  optimizer=optimizer)

	def build_generator(self):
		"""U-Net Generator"""

		def conv2d(layer_input, filters, f_size=4, bn=True):
			"""Layers used during downsampling"""
			d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
			if bn:
				d = BatchNormalization(momentum=0.8)(d)
			d = LeakyReLU(alpha=0.2)(d)
			return d

		def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0.4):
			"""Layers used during upsampling"""
			u = UpSampling2D(size=2)(layer_input)
			u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same',\
												 activation='relu')(u)
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
		output_img = Conv2D(self.channels, kernel_size=4, strides=1,\
						 padding='same', activation='tanh')(u4)

		# Model(d0, output_img).summary()
		return Model(d0, output_img)

	def build_discriminator(self):

		def d_layer(layer_input, filters, f_size=4, bn=True):
			"""Discriminator layer"""
			d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
			d = LeakyReLU(alpha=0.2)(d)
			if bn:
				d = BatchNormalization(momentum=0.8)(d)
			return d

		gt_img = Input(shape=self.img_shape)
		cond_img = Input(shape=self.img_shape)

		# Concatenate image and conditioning image by channels to produce input
		combined_imgs = Concatenate(axis=-1)([gt_img, cond_img])

		d1 = d_layer(combined_imgs, self.df, bn=False)
		d2 = d_layer(d1, self.df*2)
		d3 = d_layer(d2, self.df*4)
		d4 = d_layer(d3, self.df*8)
		d5 = d_layer(d4, self.df*16)
		d6 = d_layer(d5, self.df*16)

		validity_local = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
		validity_global = Conv2D(1, kernel_size=4, strides=1, padding='same')(d6)

		# Model([gt_img, cond_img], [validity_local , validity_global]).summary()
		return Model([gt_img, cond_img], [validity_local,validity_global])

	def train(self, epochs, batch_size=64, sample_interval=500):

		start_time = datetime.datetime.now()

		# add divyansh's dataloader functions

		data_dir = 'datasets/Capitals64/train/'
		files = os.listdir(data_dir)
		gt_imgs = []
		training_files = []

		self.block_size = 50		

		for count, file in enumerate(files):
			img = cv2.imread(data_dir + file , 0)
			new_img = []
			for i in range(26):
				new_img.append(img[:,64*i:64*(i+1)])
			new_img = np.array(new_img)
			new_img = np.transpose(new_img , (1,2,0))
			gt_imgs.append(new_img)
			training_files.append(file)
			if count>=self.block_size:
				break
		np.save('training_files.npy',training_files)

		gt_imgs = np.array(gt_imgs)
		gt_imgs = ( gt_imgs.astype( np.float32 ) - 127.5 ) / 127.5

		# Adversarial loss ground truths
		valid_local = np.ones((batch_size,) + self.disc_patch_local)
		valid_global = np.ones((batch_size,) + self.disc_patch_global)
		fake_local = np.zeros((batch_size,) + self.disc_patch_local)
		fake_global = np.zeros((batch_size,) + self.disc_patch_global)


		log_path = 'graphs/glyph/'
		callback = TensorBoard(log_path)
		callback.set_model(self.combined)
		train_names = ['dloss_local','dloss_global',\
		'gloss_local','gloss_global','gloss_L1']

		for epoch in range(epochs):
			idx = np.random.randint(0, gt_imgs.shape[0], batch_size)
			gt_imgs_batch = gt_imgs[idx]
			cond_imgs_batch = get_condition_image(gt_imgs_batch)

			# ---------------------
			#  Train Discriminator
			# ---------------------

			# Condition on B and generate a translated version
			fake_img_batch = self.generator.predict(cond_imgs_batch)

			# Train the discriminators (original images = real / generated = Fake)
			_ ,d_loss_real_local, d_loss_real_global = \
				self.discriminator.train_on_batch([gt_imgs_batch, cond_imgs_batch],\
					[valid_local,valid_global])
			_ ,d_loss_fake_local, d_loss_fake_global = \
				self.discriminator.train_on_batch([fake_img_batch, cond_imgs_batch],\
					[fake_local,fake_global])
		   
			d_loss_global = 0.5 * np.add(d_loss_real_global, d_loss_fake_global)
			d_loss_local = 0.5 * np.add(d_loss_real_local, d_loss_fake_local)

			# -----------------
			#  Train Generator
			# -----------------

			# Train the generators
			g_loss = self.combined.train_on_batch([gt_imgs_batch, cond_imgs_batch],\
			 [valid_local , valid_global, gt_imgs_batch])

			elapsed_time = datetime.datetime.now() - start_time

			write_log(callback, train_names, \
				np.asarray([d_loss_local, d_loss_global, g_loss[0], \
				 g_loss[1], g_loss[2]]), epoch)
			# Plot the progress
			print ("[Epoch %d] [D loss0: %f, D loss1: %f]\
			 [G loss0: %f , G loss1: %f , G loss2: %f] time: %s" % (epoch,d_loss_local,\
			  d_loss_global, g_loss[1],g_loss[2],g_loss[3],str(elapsed_time)))

			if epoch % 500 == 0:
				self.sample_images(epoch, cond_imgs_batch, gt_imgs_batch)

			if epoch % sample_interval == 0:
				self.save_model(epoch)



	def sample_images(self, epoch, cond_imgs, gt_imgs):
		data_dir = 'results/glyph/'+str(self.block_size)+'/'
		os.makedirs(data_dir, exist_ok=True)

		fake_imgs = self.generator.predict(cond_imgs)
		for i in range(cond_imgs.shape[0]):
			row, col = 3, cond_imgs.shape[3]

			cond_img = cond_imgs[i]
			gt_img = gt_imgs[i]
			fake_img = fake_imgs[i]

			# Rescale images 0 - 1
			fake_img = 0.5*fake_img + 0.5
			cond_img = 0.5*cond_img + 0.5
			gt_img = 0.5*gt_img + 0.5

			titles = ['Condition', 'Generated', 'Original']
			fig, axs = plt.subplots(row, col)
			for r in range(row):
				for c in range(col):
					if r==0:
						axs[r,c].imshow(cond_img[:,:,c])
					if r==1:
						axs[r,c].imshow(fake_img[:,:,c])
					if r==2:
						axs[r,c].imshow(gt_img[:,:,c])
					axs[r,c].axis('off')
			fig.savefig(data_dir + "%d_%d.png" % (epoch , i))
			plt.close()

	def save_model(self , epoch):

		def save(model, model_name):
			data_dir = "saved_models/glyph_net/"+"/"+str(epoch)
			os.makedirs(data_dir, exist_ok=True)
			model_path = data_dir + "/%s.json" % (model_name)
			weights_path = data_dir + "/%s_weights.hdf5" % (model_name)
			options = {"file_arch": model_path,
						"file_weight": weights_path}
			json_string = model.to_json()
			open(options['file_arch'], 'w').write(json_string)
			model.save_weights(options['file_weight'])

		save(self.generator, "generator")
		save(self.discriminator, "discriminator")  

glyph_model = GLYPH_MODEL()
glyph_model.train(epochs=50000)
