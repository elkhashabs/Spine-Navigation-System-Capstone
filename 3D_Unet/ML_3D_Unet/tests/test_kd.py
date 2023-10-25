import os
import shutil
import unittest

import tensorflow as tf
import numpy as np
from utils.layers import create_unet3d_class
from utils.layers_func import create_unet3d
from utils.kd import Trainer_KD


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # uncomment to not use GPU


def testKD():
    """Testing Trainer KD Class"""
    tf.random.set_seed(10)

    model_dir = 'tf_ckpt'

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    student_model = create_unet3d_class(input_shape=[64, 64, 64, 1],
                                        n_convs=1,
                                        n_filters=[2, 4],
                                        ksize=[5, 5, 5],
                                        padding='same',
                                        pooling='avg',
                                        norm='batch_norm',
                                        dropout=[0],
                                        upsampling=True,
                                        activation='relu',
                                        depth=2)

    teacher_model = create_unet3d(input_shape=[64, 64, 64, 1],
                                  n_convs=1,
                                  n_filters=[4, 8],
                                  ksize=[5, 5, 5],
                                  padding='same',
                                  pooling='avg',
                                  norm='batch_norm',
                                  dropout=[0],
                                  upsampling=True,
                                  activation='relu',
                                  depth=2)

    training_data = np.zeros([10, 64, 64, 64, 1])
    training_mask = np.zeros([10, 64, 64, 64, 1])
    validation_data = np.zeros([10, 64, 64, 64, 1])
    validation_mask = np.zeros([10, 64, 64, 64, 1])

    for j in range(10):
        xx, yy = np.mgrid[:64, :64]
        rand_x1 = np.random.randint(0, 64)
        rand_y1 = np.random.randint(0, 64)
        circle_data1 = (xx - rand_x1) ** 2 + (yy - rand_y1) ** 2
        rand_x2 = np.random.randint(0, 64)
        rand_y2 = np.random.randint(0, 64)
        circle_data2 = (xx - rand_x2) ** 2 + (yy - rand_y2) ** 2
        rand_x3 = np.random.randint(0, 64)
        rand_y3 = np.random.randint(0, 64)
        circle_data3 = (xx - rand_x3) ** 2 + (yy - rand_y3) ** 2
        rand_x4 = np.random.randint(0, 64)
        rand_y4 = np.random.randint(0, 64)
        circle_data4 = (xx - rand_x4) ** 2 + (yy - rand_y4) ** 2
        rand_x5 = np.random.randint(0, 64)
        rand_y5 = np.random.randint(0, 64)
        circle_data5 = (xx - rand_x5) ** 2 + (yy - rand_y5) ** 2
        rand_x6 = np.random.randint(0, 64)
        rand_y6 = np.random.randint(0, 64)
        circle_data6 = (xx - rand_x6) ** 2 + (yy - rand_y6) ** 2
        rand_x7 = np.random.randint(0, 64)
        rand_y7 = np.random.randint(0, 64)
        circle_data7 = (xx - rand_x7) ** 2 + (yy - rand_y7) ** 2
        circle_data = np.random.uniform(0, 1, [64, 64]) * 15 * (np.logical_and(circle_data1 < 100, circle_data1 > 0, dtype=np.float))\
                      + np.random.uniform(0, 1, [64, 64]) * 10 * (np.logical_and(circle_data2 < 150, circle_data2 > 0, dtype=np.float))\
                      + np.random.uniform(0, 1, [64, 64]) * 5 * (np.logical_and(circle_data3 < 75, circle_data3 > 0, dtype=np.float))\
                      + np.random.uniform(0, 1, [64, 64]) * 8 * (np.logical_and(circle_data4 < 125, circle_data4 > 0, dtype=np.float))\
                      + np.random.uniform(0, 1, [64, 64]) * 10 * (np.logical_and(circle_data5 < 200, circle_data5 > 0, dtype=np.float))\
                      + np.random.uniform(0, 1, [64, 64]) * 6 * (np.logical_and(circle_data6 < 250, circle_data6 > 0, dtype=np.float))\
                      + np.random.uniform(0, 1, [64, 64]) * 3 * (np.logical_and(circle_data7 < 175, circle_data7 > 0, dtype=np.float))
        training_data[j, :, :, :, 0] = np.stack((circle_data,) * 64, axis=0)
        circle_data = np.logical_and(circle_data1 < 100, circle_data1 > 0, dtype=np.float) + np.logical_and(
            circle_data4 < 125, circle_data4 > 0, dtype=np.float)
        training_mask[j, :, :, :, 0] = np.stack((circle_data,) * 64, axis=0)

    for j in range(10):
        xx, yy = np.mgrid[:64, :64]
        rand_x1 = np.random.randint(0, 64)
        rand_y1 = np.random.randint(0, 64)
        circle_data1 = (xx - rand_x1) ** 2 + (yy - rand_y1) ** 2
        rand_x2 = np.random.randint(0, 64)
        rand_y2 = np.random.randint(0, 64)
        circle_data2 = (xx - rand_x2) ** 2 + (yy - rand_y2) ** 2
        rand_x3 = np.random.randint(0, 64)
        rand_y3 = np.random.randint(0, 64)
        circle_data3 = (xx - rand_x3) ** 2 + (yy - rand_y3) ** 2
        rand_x4 = np.random.randint(0, 64)
        rand_y4 = np.random.randint(0, 64)
        circle_data4 = (xx - rand_x4) ** 2 + (yy - rand_y4) ** 2
        rand_x5 = np.random.randint(0, 64)
        rand_y5 = np.random.randint(0, 64)
        circle_data5 = (xx - rand_x5) ** 2 + (yy - rand_y5) ** 2
        rand_x6 = np.random.randint(0, 64)
        rand_y6 = np.random.randint(0, 64)
        circle_data6 = (xx - rand_x6) ** 2 + (yy - rand_y6) ** 2
        rand_x7 = np.random.randint(0, 64)
        rand_y7 = np.random.randint(0, 64)
        circle_data7 = (xx - rand_x7) ** 2 + (yy - rand_y7) ** 2
        circle_data = np.random.uniform(0, 1, [64, 64]) * 15 * (np.logical_and(circle_data1 < 100, circle_data1 > 0, dtype=np.float))\
                      + np.random.uniform(0, 1, [64, 64]) * 10 * (np.logical_and(circle_data2 < 150, circle_data2 > 0, dtype=np.float))\
                      + np.random.uniform(0, 1, [64, 64]) * 5 * (np.logical_and(circle_data3 < 75, circle_data3 > 0, dtype=np.float))\
                      + np.random.uniform(0, 1, [64, 64]) * 8 * (np.logical_and(circle_data4 < 125, circle_data4 > 0, dtype=np.float))\
                      + np.random.uniform(0, 1, [64, 64]) * 12 * (np.logical_and(circle_data5 < 200, circle_data5 > 0, dtype=np.float))\
                      + np.random.uniform(0, 1, [64, 64]) * 6 * (np.logical_and(circle_data6 < 250, circle_data6 > 0, dtype=np.float))\
                      + np.random.uniform(0, 1, [64, 64]) * 3 * (np.logical_and(circle_data7 < 175, circle_data7 > 0, dtype=np.float))
        validation_data[j, :, :, :, 0] = np.stack((circle_data,) * 64, axis=0)
        circle_data = np.logical_and(circle_data1 < 100, circle_data1 > 0, dtype=np.float) + np.logical_and(
            circle_data4 < 125, circle_data4 > 0, dtype=np.float)
        validation_mask[j, :, :, :, 0] = np.stack((circle_data,) * 64, axis=0)

    epochs = 3

    trainer = Trainer_KD(student_model, teacher_model, 10, tf.keras.optimizers.Adam, 1e-3, model_dir)

    train_data = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(training_data, dtype=tf.float32),
                                                     tf.convert_to_tensor(training_mask, dtype=tf.float32)))

    valid_data = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(validation_data, dtype=tf.float32),
                                                     tf.convert_to_tensor(validation_mask, dtype=tf.float32)))

    train_ds = tf.distribute.get_strategy().experimental_distribute_dataset(train_data.batch(1).repeat())
    valid_ds = tf.distribute.get_strategy().experimental_distribute_dataset(valid_data.batch(1).repeat())

    loss_fn = tf.keras.losses.binary_crossentropy
    accuracy_fn = tf.keras.metrics.binary_accuracy

    trainer.train(train_ds=train_ds, valid_ds=valid_ds, train_size=10, validation_size=10, loss_fn=loss_fn,
                  distillation_fn=tf.keras.losses.binary_crossentropy, accuracy_fn=accuracy_fn, BATCH_SIZE=1,
                  EPOCHS=epochs, save_step=1)

    shutil.rmtree(model_dir)
    shutil.rmtree(model_dir + '_student_scratch')
    shutil.rmtree(model_dir + '_teacher_scratch')


if __name__ == '__main__':
    unittest.main()
