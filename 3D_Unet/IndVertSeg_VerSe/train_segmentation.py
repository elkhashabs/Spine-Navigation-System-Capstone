import tensorflow as tf
import sys
sys.path.insert(1, 'F:\\Xiangping\\498')

from utils.train_segmentation_utils import DatasetHandler, Trainer
from ML_3D_Unet.utils.layers_func import create_unet3d

training_directory = 'Data/Training/'
validation_directory = 'Data/Validation/'
training_size = 2814
validation_size = 1494
batch_size = 1 * tf.distribute.get_strategy().num_replicas_in_sync
datasets = DatasetHandler(training_directory, validation_directory, training_size, validation_size, batch_size)
train_data, valid_data = datasets.get_datasets()

with tf.distribute.get_strategy().scope():
    model = create_unet3d([128, 128, 128, 2], n_convs=2, n_filters=[16, 32, 64, 128], ksize=[3, 3, 3], padding='same',
                          activation='relu', pooling='max', norm='batch_norm', dropout=[0], depth=4, upsampling=True)
model.summary()

lr = 1e-3
optimizer = tf.keras.optimizers.Adam
model_dir = 'tf_ckpt_teacher_scratch'
n_epochs = 45

print('\nTraining Segmentation Network')

trainer = Trainer(model, optimizer, lr, model_dir)
trainer.train(train_ds=train_data, valid_ds=valid_data, train_size=training_size, validation_size=validation_size,
              BATCH_SIZE=batch_size, EPOCHS=45)
