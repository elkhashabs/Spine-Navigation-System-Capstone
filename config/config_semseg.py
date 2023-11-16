import os
from config.paths import train_images_folder, train_labels_folder, train_images, \
    train_labels, val_images_folder, val_labels_folder, val_images, \
    val_labels
from semseg.data_loader import SemSegConfig


class SemSegMRIConfig(SemSegConfig):
    train_images = sorted([os.path.join(train_images_folder, train_image)
                           for train_image in train_images])
    train_labels = sorted([os.path.join(train_labels_folder, train_label)
                           for train_label in train_labels])
    val_images = sorted([os.path.join(val_images_folder, val_image)
                           for val_image in val_images])
    val_labels = sorted([os.path.join(val_labels_folder, val_label)
                           for val_label in val_labels])
    do_normalize = False
    batch_size = 8
    num_workers = 8
    lr = 0.0001
    # lr = 0.005
    epochs = 1000
    # epochs = 500
    low_lr_epoch = epochs // 10
    val_epochs = epochs // 10
    cuda = True
    num_outs = 2
    do_crossval = False
    num_folders = 2
    num_channels = 8
