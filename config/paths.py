import os
import sys

logs_folder = "logs"
os.makedirs(logs_folder, exist_ok=True)
current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)

base_dataset_dir = 'F:\\Xiangping\\498\\osfstorage-archive'

# training
train_images_folder = os.path.join(base_dataset_dir, "training_c")
train_labels_folder = os.path.join(base_dataset_dir, "training_mask_c")

os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)

original_train_images_folder = os.path.join(base_dataset_dir, "dataset-verse19training\\images")
original_train_labels_folder = os.path.join(base_dataset_dir, "dataset-verse19training\\labels")
train_prediction_folder = os.path.join(base_dataset_dir, "predTr")
multiclass_prediction_folder = os.path.join(base_dataset_dir, "predMulticlass")
train_images = os.listdir(train_images_folder)
train_labels = os.listdir(train_labels_folder)
original_train_images = os.listdir(original_train_images_folder)
original_train_labels = os.listdir(original_train_labels_folder)

train_images = [train_image for train_image in train_images
                if train_image.endswith(".nii.gz") and not train_image.startswith('.')]
train_labels = [train_label for train_label in train_labels
                if train_label.endswith(".nii.gz") and not train_label.startswith('.')]
original_train_images = [train_image for train_image in original_train_images
                if train_image.endswith(".nii.gz") and not train_image.startswith('.')]
original_train_labels = [train_label for train_label in original_train_labels
                if train_label.endswith(".nii.gz") and not train_label.startswith('.')]

# validation 
val_images_folder = os.path.join(base_dataset_dir, "validation_c")
val_labels_folder = os.path.join(base_dataset_dir, "validation_mask_c")

os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

original_val_images_folder = os.path.join(base_dataset_dir, "dataset-verse19validation\\images")
original_val_labels_folder = os.path.join(base_dataset_dir, "dataset-verse19validation\\labels")
val_prediction_folder = os.path.join(base_dataset_dir, "predTr")
multiclass_prediction_folder = os.path.join(base_dataset_dir, "predMulticlass")
val_images = os.listdir(val_images_folder)
val_labels = os.listdir(val_labels_folder)
original_val_images = os.listdir(original_val_images_folder)
original_val_labels = os.listdir(original_val_labels_folder)

val_images = [val_image for val_image in val_images
                if val_image.endswith(".nii.gz") and not val_image.startswith('.')]
val_labels = [val_label for val_label in val_labels
                if val_label.endswith(".nii.gz") and not val_label.startswith('.')]
original_val_images = [val_image for val_image in original_val_images
                if val_image.endswith(".nii.gz") and not val_image.startswith('.')]
original_val_labels = [val_label for val_label in original_val_labels
                if val_label.endswith(".nii.gz") and not val_label.startswith('.')]

# testing
test_images_folder = "F:\\Xiangping\\498\\AXIAL\\test\\test"
os.makedirs(test_images_folder, exist_ok=True)
test_images = os.listdir(test_images_folder)
test_prediction_folder = os.path.join(base_dataset_dir, "predTs")

test_images = [test_image for test_image in test_images
               if test_image.endswith(".nii.gz") and not test_image.startswith('.')]

labels_names = {
   "0": "background",
   "1": "Spine",
 }
labels_names_list = [labels_names[el] for el in labels_names]
