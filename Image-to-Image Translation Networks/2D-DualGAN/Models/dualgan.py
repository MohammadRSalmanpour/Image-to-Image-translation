# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 23:47:22 2024

@author: Amin
"""
from __future__ import print_function, division
import scipy
import os

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

def build_generator(input_shape):

    img_input = Input(shape=input_shape)

    x = Dense(256)(img_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dropout(0.4)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dropout(0.4)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dropout(0.4)(x)
    x = Dense(1, activation='tanh')(x) # input_shape[0] * input_shape[1]

    X_translated = Model(img_input, x)

    return X_translated

def build_discriminator(input_shape):

    img_input = Input(shape=input_shape)
    
    x = Dense(512)(img_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(1)(x)

    validity = Model(img_input, x)

    return validity

class DUALGAN():
    def __init__(self, shape, generatorAB, generatorBA, discriminatorDA, discriminatorDB):
        self.img_rows = shape[0]
        self.img_cols = shape[1]
        self.channels = shape[2]
        self.input_shape = shape

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.D_A = discriminatorDA
        self.D_A.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])
        self.D_B = discriminatorDB
        self.D_B.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.G_AB = generatorAB
        self.G_BA = generatorBA

        # For the combined model we will only train the generators
        self.D_A.trainable = False
        self.D_B.trainable = False

        # The generator takes images from their respective domains as inputs
        imgs_A = Input(shape=self.input_shape)
        imgs_B = Input(shape=self.input_shape)

        # Generators translates the images to the opposite domain
        fake_B = self.G_AB(imgs_A)
        fake_A = self.G_BA(imgs_B)

        # The discriminators determines validity of translated images
        valid_A = self.D_A(fake_A)
        valid_B = self.D_B(fake_B)

        # Generators translate the images back to their original domain
        recov_A = self.G_BA(fake_B)
        recov_B = self.G_AB(fake_A)

        # The combined model  (stacked generators and discriminators)
        self.combined = Model(inputs=[imgs_A, imgs_B], outputs=[valid_A, valid_B, recov_A, recov_B])
        self.combined.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, 'mae', 'mae'],
                            optimizer=optimizer,
                            loss_weights=[1, 1, 100, 100])

    def sample_generator_input(self, X, batch_size):
        # Sample random batch of images from X
        idx = np.random.randint(0, X.shape[0], batch_size)
        return X[idx]

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def train(self, train_dataset, example_input, example_target, sample_generated_image_path, last_models_dir, epochs=1000, batch_size=1):
        
        example_input = np.expand_dims(example_input, axis=0)
        example_target = np.expand_dims(example_target, axis=0)
        
        clip_value = 0.01
        n_critic = 4

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        # Losses Lists
        g_loss_list = []
        D_A_loss_list = []
        D_B_loss_list = []
        
        for epoch in range(epochs):
            
            for imgs_A, imgs_B in train_dataset.batch(batch_size): 
                # Train the discriminator for n_critic iterations
                for _ in range(n_critic):
    
                    # ----------------------
                    #  Train Discriminators
                    # ----------------------
    
                    # Sample generator inputs
                    #imgs_A = self.sample_generator_input(X_A, batch_size)
                    #imgs_B = self.sample_generator_input(X_B, batch_size)
    
                    # Translate images to their opposite domain
                    fake_B = self.G_AB.predict(imgs_A)
                    fake_A = self.G_BA.predict(imgs_B)
    
                    # Train the discriminators
                    D_A_loss_real = self.D_A.train_on_batch(imgs_A, valid)
                    D_A_loss_fake = self.D_A.train_on_batch(fake_A, fake)
    
                    D_B_loss_real = self.D_B.train_on_batch(imgs_B, valid)
                    D_B_loss_fake = self.D_B.train_on_batch(fake_B, fake)
    
                    D_A_loss = 0.5 * np.add(D_A_loss_real, D_A_loss_fake)
                    D_B_loss = 0.5 * np.add(D_B_loss_real, D_B_loss_fake)
    
                    # Clip discriminator weights
                    for d in [self.D_A, self.D_B]:
                        for l in d.layers:
                            weights = l.get_weights()
                            weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                            l.set_weights(weights)
    
                # ------------------
                #  Train Generators
                # ------------------
    
                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B])
    
            # Plot the progress
            print ("Epoch %d: [D1 loss: %f] [D2 loss: %f] [G loss: %f]" \
                % (epoch + 1, D_A_loss[0], D_B_loss[0], g_loss[0]))
    

            # saved models
            self.G_AB.save_weights(os.path.join(last_models_dir, 'Generator_AB.h5'))
            self.G_BA.save_weights(os.path.join(last_models_dir, 'Generator_BA.h5'))
            self.D_A.save_weights(os.path.join(last_models_dir, 'Discriminator_D_A.h5'))
            self.D_B.save_weights(os.path.join(last_models_dir, 'Discriminator_D_B.h5'))
            
            # save models for each epoch
            dst_path = os.path.join(last_models_dir, 'Epoch{}'.format(epoch))
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            
            self.G_AB.save_weights(os.path.join(dst_path, 'Generator_AB.h5'))
            self.G_BA.save_weights(os.path.join(dst_path, 'Generator_BA.h5'))
            self.D_A.save_weights(os.path.join(dst_path, 'Discriminator_D_A.h5'))
            self.D_B.save_weights(os.path.join(dst_path, 'Discriminator_D_B.h5'))
            
            # save a sample generated image
            sample_generated_image_path_epoch = os.path.join(sample_generated_image_path, f'SampleGenerated_epoch{epoch + 1}')
            self.generate_images(self.G_AB, example_input, example_target, sample_generated_image_path_epoch)
            
            # save losses plot
            g_loss_list.append(g_loss[0])
            D_A_loss_list.append(D_A_loss[0])
            D_B_loss_list.append(D_B_loss[0])
            
            epochs_range = range(1, len(g_loss_list) + 1)
            plt.plot(epochs_range, g_loss_list, label='G_loss')
            plt.plot(epochs_range, D_A_loss_list, label='D_A_loss')
            plt.plot(epochs_range, D_B_loss_list, label='D_B_loss')
            plt.title('Generator and Discriminator loss in Epoch {}'.format(epoch))
            plt.legend()
            plt.savefig(os.path.join(dst_path, 'loss_graph_in_epoch {}'.format(epoch)))
            plt.close()

    def generate_images(self, model, test_input, tar, save_to):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
          plt.subplot(1, 3, i+1)
          plt.title(title[i])
          # Getting the pixel values in the [0, 1] range to plot.
          plt.imshow(display_list[i] * 0.5 + 0.5)
          plt.axis('off')
        plt.savefig(save_to)
        plt.close()


"""if __name__ == '__main__':
    gan = DUALGAN()
    gan.train(epochs=1000, batch_size=1)"""
