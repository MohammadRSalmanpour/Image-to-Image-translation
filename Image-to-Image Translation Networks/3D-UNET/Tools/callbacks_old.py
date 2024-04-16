from tensorflow.keras.callbacks import Callback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class SaveModel(Callback):
    def __init__(self, save_best_model_to, save_last_model_to):
        super(SaveModel, self).__init__()
        self.best_model_path = save_best_model_to
        self.last_model_path = save_last_model_to
        self.best_loss = float('inf')
        self.epoch_number = 1
        
    def on_train_begin(self, logs={}):
        self.history = {'loss': [], 'val_loss': []}

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.model.save_weights(self.best_model_path, overwrite=True)
            print(f"Best model weights saved with validation loss: {val_loss:.4f}")
        self.model.save_weights(self.last_model_path, overwrite=True)
        # print(f"Last model weights saved with validation loss: {val_loss:.4f}")
        
        dst_path = os.path.join(self.last_model_path[:-17], 'Epoch {}'.format(self.epoch_number))
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        self.model.save_weights(os.path.join(dst_path, 'EncoderDecoder.h5'), overwrite=True)

        self.history['loss'].append(logs['loss'])
        self.history['val_loss'].append(logs['val_loss'])


        epochs = range(1, len(self.history['loss']) + 1)
        plt.plot(epochs, self.history['loss'], label='Train loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation loss')
        plt.title('Train and Validation loss in Epoch {}'.format(self.epoch_number))
        plt.legend()
        plt.savefig(os.path.join(dst_path, 'AutoEncoder_loss_graph_in_epoch {}'.format(self.epoch_number)))
        plt.close()
        self.epoch_number += 1
