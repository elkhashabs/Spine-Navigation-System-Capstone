import os
import shutil

training_images = "F:\\Xiangping\\498\\osfstorage-archive\\dataset-verse19training\\rawdata"
training_labels = "F:\\Xiangping\\498\\osfstorage-archive\\dataset-verse19training\\derivatives"
# validation_images = "F:\\Xiangping\\498\\osfstorage-archive\\dataset-verse19validation\\rawdata"
# validation_labels = "F:\\Xiangping\\498\\osfstorage-archive\\dataset-verse19validation\\derivatives"

train_image_des = "F:\\Xiangping\\498\\osfstorage-archive\\dataset-verse19training\\images"
train_label_des = "F:\\Xiangping\\498\\osfstorage-archive\\dataset-verse19training\\labels"
# val_image_des = "F:\\Xiangping\\498\\osfstorage-archive\\dataset-verse19validation\\images"
# val_label_des = "F:\\Xiangping\\498\\osfstorage-archive\\dataset-verse19validation\\labels"
os.makedirs(train_image_des, exist_ok=True)
os.makedirs(train_label_des, exist_ok=True)
# os.makedirs(val_image_des, exist_ok=True)
# os.makedirs(val_label_des, exist_ok=True)

# training
train_image = []
for images in os.listdir(training_images):
    for image in os.listdir(training_images+"/"+images):
        if image.endswith('.gz'):
            train_image.append(image)
            shutil.move(training_images+"/"+images+"/"+image, train_image_des)
     
train_labels = []
for labels in os.listdir(training_labels):
    for label in os.listdir(training_labels+"/"+labels):
        if label.endswith('.gz'):
            train_labels.append(label)
            shutil.move(training_labels+"/"+labels+"/"+label, train_label_des)

# validation
# val_images = []
# for images in os.listdir(validation_images):
#     for image in os.listdir(validation_images+"/"+images):
#         if image.endswith('.gz'):
#             val_images.append(image)
#             shutil.move(validation_images+"/"+images+"/"+image, val_image_des)

# val_labels = []
# for labels in os.listdir(validation_labels):
#     for label in os.listdir(validation_labels+"/"+labels):
#         if label.endswith('.gz'):
#             val_labels.append(label)
#             shutil.move(validation_labels+"/"+labels+"/"+label, val_label_des)