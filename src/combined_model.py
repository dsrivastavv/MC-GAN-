from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Multiply, add, Concatenate
from keras.layers import BatchNormalization, Activation, Lambda, Embedding
from keras.layers import ZeroPadding2D, DepthwiseConv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D , Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from scipy.misc import toimage, imsave
import keras
import datetime
import keras.backend as K
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pdb
import os
import cv2
import random

def generate_inputs(img, img_color, num_letters):

	idx = random.sample(range(0, 26), num_letters)  # the indices to keep
	batch_cond_img = np.ones((num_letters+1, 64, 64, 26)) # input of glyph network
	batch_mask_img = np.zeros((num_letters+1, 64, 64, 26))
	weight_mask_img = np.ones((64, 64, 26))

	for i in range(num_letters):
		batch_cond_img[i, :, :, :][:, :, idx] = img[:, :, idx]
		batch_cond_img[i, :, :, :][:, :, idx[i]] = np.ones((64, 64))
		batch_mask_img[i, :, :, idx[i]] = np.ones((64,64))
		weight_mask_img[:,:,idx[i]] = 10*np.ones((64,64))

	batch_cond_img[-1, :, :, :][:, :, idx] = img[:, :, idx]
	batch_mask_img[-1, :, :, :][:, :, :] = np.ones((64, 64, 26))
	batch_mask_img[-1, :, :, :][:, :, idx] = np.zeros((64, 64, num_letters))

	gt_color_img = np.transpose(img_color, (1, 2, 0, 3))

	return gt_color_img, batch_cond_img, batch_mask_img, weight_mask_img, idx

def write_log(callback, names, logs, batch_no):
	for name, value in zip(names, logs):
		summary = tf.Summary()
		summary_value = summary.value.add()
		summary_value.simple_value = value
		summary_value.tag = name
		callback.writer.add_summary(summary, batch_no)
		callback.writer.flush()

class COMBINED_MODEL():
	def __init__(self):
		# Input shape
		self.img_rows = 64
		self.img_cols = 64
		self.channels = 26
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.img_color_shape = (self.img_rows, self.img_cols, self.channels, 3)
		# # Calculate output shape of D (PatchGAN)
		# Number of filters in the first layer of G and D
		self.gf = 64
		self.df = 52
		lr = 0.0002
		optimizer = Adam(lr*0.00001, 0.5)
		self.block_size = 50

		glyph_model_path = 'saved_models/glyph_'+str(self.block_size)+'/6500/generator.json'
		glyph_weights_path = 'saved_models/glyph_'+str(self.block_size)+'/6500/generator_weights.hdf5'
		self.glyph_generator = 	self.load_model(glyph_model_path, glyph_weights_path, 'glyph_generator')
		self.original_glyph_generator = self.load_model(glyph_model_path, glyph_weights_path, 'original_glyph_generator')
		self.original_glyph_generator.trainable = False


		orna_model_path = 'saved_models/orna_net/'+ '/1500/'
		self.orna_generator = self.load_model(orna_model_path+'generator_orna.json', orna_model_path+'generator_orna_weights.hdf5', 'orna_generator')
		self.orna_discriminator = self.load_model(orna_model_path+'discriminator_orna.json', orna_model_path+'discriminator_orna_weights.hdf5', 'orna_discriminator')

		self.orna_discriminator.compile(loss=['mse','mse'],
			optimizer=optimizer)

		self.num_letters = 5 # input will always have only 5 letters

		# define output shape of discriminator
		patch_local = 4
		patch_global = 1
		self.disc_patch_local = (patch_local, patch_local, self.channels)
		self.disc_patch_global = (patch_global, patch_global, self.channels)

		self.disc_patch_local_train = (patch_local, patch_local, 1)
		self.disc_patch_global_train = (patch_global, patch_global, 1)

		#-------------------------
		# Construct Computational
		#   Graph of Generator
		#-------------------------

		# Input images and their conditioning images
		gt_color_img = Input(shape=self.img_color_shape)

		# leave one out conditional images
		cond_img1 = Input(shape=self.img_shape)
		cond_img2 = Input(shape=self.img_shape)
		cond_img3 = Input(shape=self.img_shape)
		cond_img4 = Input(shape=self.img_shape)
		cond_img5 = Input(shape=self.img_shape)
		cond_img6 = Input(shape=self.img_shape)

		mask_img1 = Input(shape=self.img_shape) #mask is [64, 64, 26] with everything other than corresponding letter zeroed out
		mask_img2 = Input(shape=self.img_shape)
		mask_img3 = Input(shape=self.img_shape)
		mask_img4 = Input(shape=self.img_shape)
		mask_img5 = Input(shape=self.img_shape)
		mask_img6 = Input(shape=self.img_shape)

		weight_mask = Input(shape=self.img_shape) # for the L1 losss between pretrained glyph output and new one

		glyph_out_img1 = self.glyph_generator(cond_img1)
		glyph_out_img2 = self.glyph_generator(cond_img2)
		glyph_out_img3 = self.glyph_generator(cond_img3)
		glyph_out_img4 = self.glyph_generator(cond_img4)
		glyph_out_img5 = self.glyph_generator(cond_img5)
		glyph_out_img6 = self.glyph_generator(cond_img6)

		glyph_letter1 = Multiply()([glyph_out_img1,mask_img1])
		glyph_letter2 = Multiply()([glyph_out_img2,mask_img2])
		glyph_letter3 = Multiply()([glyph_out_img3,mask_img3])
		glyph_letter4 = Multiply()([glyph_out_img4,mask_img4])
		glyph_letter5 = Multiply()([glyph_out_img5,mask_img5])
		glyph_letter_rem = Multiply()([glyph_out_img6,mask_img6])
		sigmoid_layer = Activation('sigmoid')

		glyph_letters_all = add([glyph_letter1, glyph_letter2, glyph_letter3, glyph_letter4, glyph_letter5, glyph_letter_rem])
		# repeat-vector 
		orna_inp = Lambda(lambda x : tf.tile(tf.expand_dims(x, axis=-1), [1,1,1,1,3]), name='orna_inp_lb')(glyph_letters_all)

		valid_local = None
		valid_global = None
		generated_color_img = None
		masked_color_img = None

		for i in range(26):
			orna_alp = Lambda( lambda x : x[:,:,:,i,:], name='orna_alp_lb'+str(i))(orna_inp) 
			generated_color_alp = self.orna_generator(orna_alp)
			masked_color_alp = sigmoid_layer(generated_color_alp)
			valid_local_alp , valid_global_alp = self.orna_discriminator([generated_color_alp, orna_alp])
			generated_color_alp = Lambda( lambda x: tf.expand_dims(x, axis=-1), name='generated_color_alp_lb'+str(i))(generated_color_alp)
			masked_color_alp = Lambda( lambda x: tf.expand_dims(x, axis=-1), name='masked_color_alp_lb'+str(i))(masked_color_alp)
			if i == 0:
				valid_local = valid_local_alp
				valid_global = valid_global_alp
				generated_color_img = generated_color_alp
				masked_color_img = masked_color_alp
			else:
				valid_local = Concatenate(axis=-1)([valid_local, valid_local_alp]) #shape=(None, 4, 4, 26)
				valid_global = Concatenate(axis=-1)([valid_global, valid_global_alp])
				generated_color_img = Concatenate(axis=-1)([generated_color_img, generated_color_alp]) #shape=(None, 64, 64, 3, 26)
				masked_color_img = Concatenate(axis=-1)([masked_color_img, masked_color_alp])

		self.orna_discriminator.trainable = False

		generated_color_img = Lambda( lambda x: tf.transpose(x, [0,1,2,4,3]))(generated_color_img)
		masked_color_img = Lambda( lambda x: tf.transpose(x, [0,1,2,4,3]))(masked_color_img)

		weighted_glyph_letters_all = Multiply()([weight_mask, glyph_letters_all])

		self.combined = Model(inputs=[gt_color_img, cond_img1,cond_img2, cond_img3, cond_img4, cond_img5, cond_img6, mask_img1, mask_img2,mask_img3, mask_img4, mask_img5, mask_img6, weight_mask ], outputs=[valid_local, valid_global, generated_color_img, masked_color_img, weighted_glyph_letters_all, masked_color_img])

		self.combined.compile(loss=['mse', 'mse', 'mae', 'mse', 'mae', 'mse'],
							  loss_weights=[1, 10, 300, 300, 10, 300],
							  optimizer=optimizer)

	def train(self, epochs, batch_size=1, sample_interval=10):

		start_time = datetime.datetime.now()

		data_dir = 'datasets/Capitals_colorGrad64/train/'
		files = os.listdir(data_dir)
		gt_imgs = []
		gt_imgs_color = []				

		self.block_size = 50
		for count, file in enumerate(files):
			img = cv2.imread(data_dir + file , 0)
			img_color = cv2.imread(data_dir + file)
			new_img = []
			new_img_color = []
			for i in range(26):
				new_img.append(img[:,64*i:64*(i+1)])
				new_img_color.append(img_color[:, 64*i:64*(i+1)])
			new_img = np.array(new_img)
			new_img_color = np.array(new_img_color)
			new_img = np.transpose(new_img , (1,2,0))
			gt_imgs.append(new_img)
			gt_imgs_color.append(new_img_color)
			if count>=self.block_size:
				break
		gt_imgs = np.array(gt_imgs)
		gt_imgs = ( gt_imgs.astype( np.float32 ) - 127.5 ) / 127.5 #shape = (B, 64, 64, 26)

		gt_imgs_color = np.array(gt_imgs_color)
		gt_imgs_color = (gt_imgs_color.astype(np.float32) - 127.5) / 127.5  #shape = (B, 26, 64, 64, 3)

		# Adversarial loss ground truths 
		valid_local = np.ones((batch_size,) + self.disc_patch_local) #shape = (1, 4, 4, 26)
		valid_global = np.ones((batch_size,) + self.disc_patch_global)


		disc_valid_local = np.ones((26,) + self.disc_patch_local_train)
		disc_valid_global = np.ones((26,) + self.disc_patch_global_train)
		disc_fake_local = np.zeros((26,) + self.disc_patch_local_train)
		disc_fake_global = np.zeros((26,) + self.disc_patch_global_train)


		log_path = 'graphs/combined/'
		callback = TensorBoard(log_path)
		callback.set_model(self.combined)
		train_names = ['dloss_local','dloss_global',\
		'gloss_local','gloss_global','gloss_L1']

		for epoch in range(epochs):
			for font_num in range(gt_imgs_color.shape[0]):
				idx = np.random.randint(0, gt_imgs.shape[0], 1)

				gt_color_img, batch_cond_img, batch_mask_img, weight_mask_img,  img_indices = \
				generate_inputs(gt_imgs[idx, :, :, :][0], gt_imgs_color[idx, :, :, :, :][0], self.num_letters)
				gt_color_img_batch = np.expand_dims(gt_color_img, axis=0)
				
				cond_img1_batch = np.expand_dims(batch_cond_img[0, :, :, :], axis=0)
				cond_img2_batch = np.expand_dims(batch_cond_img[1, :, :, :], axis=0)
				cond_img3_batch = np.expand_dims(batch_cond_img[2, :, :, :], axis=0)
				cond_img4_batch = np.expand_dims(batch_cond_img[3, :, :, :], axis=0)
				cond_img5_batch = np.expand_dims(batch_cond_img[4, :, :, :], axis=0)
				cond_img6_batch = np.expand_dims(batch_cond_img[5, :, :, :], axis=0)

				mask_img1_batch = np.expand_dims(batch_mask_img[0, :, :, :], axis=0)
				mask_img2_batch = np.expand_dims(batch_mask_img[1, :, :, :], axis=0)
				mask_img3_batch = np.expand_dims(batch_mask_img[2, :, :, :], axis=0)
				mask_img4_batch = np.expand_dims(batch_mask_img[3, :, :, :], axis=0)
				mask_img5_batch = np.expand_dims(batch_mask_img[4, :, :, :], axis=0)
				mask_img6_batch = np.expand_dims(batch_mask_img[5, :, :, :], axis=0)

				weight_mask_batch = np.expand_dims(weight_mask_img, axis=0)

				orig_glyph_generator_output_1 = self.original_glyph_generator.predict(batch_cond_img) #shape = (6,64,64,26)
				orig_glyph_generator_output = orig_glyph_generator_output_1[-1, :, :, :]

				for i in range(self.num_letters):
					orig_glyph_generator_output[:, :, img_indices[i]] = orig_glyph_generator_output_1[i, :, :, img_indices[i]]					
				weighted_orig_glyph_generator_output = np.expand_dims(weight_mask_img*orig_glyph_generator_output, axis=0)


				glyph_generator_output_1 = self.glyph_generator.predict(batch_cond_img)				
				glyph_generator_output = glyph_generator_output_1[-1, :, :, :]
				for i in range(self.num_letters):
					glyph_generator_output[:, :, img_indices[i]] = glyph_generator_output_1[i, :, :, img_indices[i]]					

				glyph_generator_output = np.array([glyph_generator_output, glyph_generator_output, glyph_generator_output])
				glyph_generator_output_batch = np.expand_dims(np.transpose(glyph_generator_output, (1,2,3,0)),axis=0)


				# ---------------------
				#  Train Discriminator
				# ---------------------
				glyph_generator_output_batch_for_disc = np.transpose(glyph_generator_output_batch[0,:,:,:], (2,0,1,3))
				gt_color_img_batch_for_disc = np.transpose(gt_color_img_batch[0,:,:,:], (2,0,1,3))
				fake_img_batch_for_disc = self.orna_generator.predict(glyph_generator_output_batch_for_disc)

				# # Condition on B and generate a translated version
				# fake_img_batch = np.zeros((26,1,64,64,3))
				# for i in range(26):
				# 	fake_img = self.orna_generator.predict(glyph_generator_output_batch[:,:,:,i,:])
				# 	fake_img_batch[i,:,:,:,:] = fake_img

				# fake_img_batch = np.transpose(fake_img_batch, [1,2,3,0,4])

				# Train the discriminators (original images = real / generated = Fake)

				_ ,d_loss_real_local, d_loss_real_global = \
					self.orna_discriminator.train_on_batch([gt_color_img_batch_for_disc, glyph_generator_output_batch_for_disc],\
						[disc_valid_local, disc_valid_global])
				_ ,d_loss_fake_local, d_loss_fake_global = \
					self.orna_discriminator.train_on_batch([fake_img_batch_for_disc, glyph_generator_output_batch_for_disc],\
						[disc_fake_local,disc_fake_global])
				
				d_loss_global = 0.5 * np.add(d_loss_real_global, d_loss_fake_global)
				d_loss_local = 0.5 * np.add(d_loss_real_local, d_loss_fake_local)

				# Train the combined_model
				g_loss = self.combined.train_on_batch([gt_color_img_batch, \
					cond_img1_batch,cond_img2_batch, cond_img3_batch, cond_img4_batch, \
					cond_img5_batch, cond_img6_batch, mask_img1_batch, mask_img2_batch,mask_img3_batch,\
					 mask_img4_batch, mask_img5_batch, mask_img6_batch, weight_mask_batch ],\
				 [valid_local , valid_global, gt_color_img_batch, 1/(1+np.exp(-glyph_generator_output_batch)), weighted_orig_glyph_generator_output, 1/(1+np.exp(-gt_color_img_batch))])

				elapsed_time = datetime.datetime.now() - start_time

				write_log(callback, train_names, \
					np.asarray([d_loss_local, d_loss_global, g_loss[0], \
					 g_loss[1], g_loss[2]]), epoch)
				# Plot the progress
				print ("[Epoch %d] [D loss0: %f, D loss1: %f]\
				 [G loss0: %f , G loss1: %f , G loss2: %f , G loss3: %f , G loss4: %f , G loss5: %f , G loss6: %f] time: %s" % (epoch, d_loss_local,\
				  d_loss_global, g_loss[0], g_loss[1], g_loss[2], g_loss[3], g_loss[4], g_loss[5], g_loss[6], str(elapsed_time)))

				if epoch % 10 == 0:
					print('writing images')
					self.sample_images(epoch, font_num, glyph_generator_output_batch, gt_color_img_batch)
			if epoch % sample_interval == 0:
				self.save_model(epoch)



	def sample_images(self, epoch, font_num, cond_imgs, gt_imgs):
		# cond_imgs is condition for orna net and gt_imgs is colored gt image
		output_dir = 'results/combined/'+str(epoch)
		os.makedirs(output_dir, exist_ok=True)

		fake_imgs = np.zeros((26,1,64,64,3))
		for i in range(26):
			fake_img = self.orna_generator.predict(cond_imgs[:,:,:,i,:])
			fake_imgs[i,:,:,:,:] = fake_img

		fake_imgs = np.transpose(fake_imgs, [1,2,3,0,4])

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
						axs[r,c].imshow(cond_img[:,:,c,:])

					if r==1:
						axs[r,c].imshow(fake_img[:,:,c,:])

					if r==2:
						axs[r,c].imshow(gt_img[:,:,c,:])
						
					axs[r,c].axis('off')
			fig.savefig(output_dir + "%d_%d_%d.png" % (epoch , i, font_num))
			plt.close()

	def save_model(self , epoch):

		def save(model, model_name):
			data_dir = "saved_models/combined_net"+str(self.block_size)+"/"+str(epoch)
			os.makedirs(data_dir, exist_ok=True)
			model_path = data_dir + "/%s.json" % (model_name)
			weights_path = data_dir + "/%s_weights.hdf5" % (model_name)
			options = {"file_arch": model_path,
						"file_weight": weights_path}
			json_string = model.to_json()
			open(options['file_arch'], 'w').write(json_string)
			model.save_weights(options['file_weight'])

		save(self.glyph_generator, "glyph_generator")
		save(self.orna_generator, "generator_orna")
		save(self.orna_discriminator, "discriminator_orna")  


	def load_model(self, model_path, weights_path, model_name):
		json_file = open(model_path, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = keras.models.model_from_json(loaded_model_json)
		model.name = model_name
		model.load_weights(weights_path)

		return model

combined_model = COMBINED_MODEL()
combined_model.train(epochs=50000)


