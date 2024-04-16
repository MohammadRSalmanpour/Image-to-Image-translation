from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np

class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, example_us, example_mri, args):
        self.example_input = np.expand_dims(example_us, axis=0)
        self.example_target = np.expand_dims(example_mri, axis=0)
        self.args = args
        self.epoch_number = 1
    
    def on_train_begin(self, logs={}):
        self.history={'G_loss': [],'F_loss': [],'D_X_loss': [],'D_Y_loss': []}   

    def on_epoch_end(self, epoch, logs=None):
        # save a sample generated image
        sample_generated_image_path = os.path.join(self.args['sample_generated_images_dir'], f'SampleGenerated_epoch{self.epoch_number}')
        generate_images(self.model.gen_G, self.example_input, self.example_target, sample_generated_image_path)

        # saved models
        self.model.gen_G.save_weights(os.path.join(self.args['last_models_dir'], 'Generator_G.h5'))
        self.model.gen_F.save_weights(os.path.join(self.args['last_models_dir'], 'Generator_F.h5'))
        self.model.disc_X.save_weights(os.path.join(self.args['last_models_dir'], 'Discriminator_X.h5'))
        self.model.disc_Y.save_weights(os.path.join(self.args['last_models_dir'], 'Discriminator_Y.h5')) 
        
        # save models for each epoch
        dst_path = os.path.join(self.args['last_models_dir'], 'Epoch{}'.format(self.epoch_number))
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        
        self.model.gen_G.save_weights(os.path.join(dst_path, 'Generator_G.h5'))
        self.model.gen_F.save_weights(os.path.join(dst_path, 'Generator_F.h5'))
        self.model.disc_X.save_weights(os.path.join(dst_path, 'Discriminator_X.h5'))
        self.model.disc_Y.save_weights(os.path.join(dst_path, 'Discriminator_Y.h5'))
        
        self.history['G_loss'].append(logs.get('G_loss'))
        self.history['F_loss'].append(logs.get('F_loss'))
        self.history['D_X_loss'].append(logs.get('D_X_loss'))
        self.history['D_Y_loss'].append(logs.get('D_Y_loss'))
        
        epochs = range(1, len(self.history['G_loss']) + 1)
        plt.plot(epochs, self.history['G_loss'], label='G_loss')
        plt.plot(epochs, self.history['F_loss'], label='F_loss')
        plt.plot(epochs, self.history['D_X_loss'], label='D_X_loss')
        plt.plot(epochs, self.history['D_Y_loss'], label='D_Y_loss')
        plt.title('Generator and Discriminator loss in Epoch {}'.format(self.epoch_number))
        plt.legend()
        plt.savefig(os.path.join(dst_path, 'loss_graph_in_epoch {}'.format(self.epoch_number)))
        plt.close()
	
        self.epoch_number += 1
      

def generate_images(model, test_input, tar, save_to):
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
