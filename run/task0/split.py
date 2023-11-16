import argparse
import os
import shutil
import sys
sys.path.insert(1, 'F:\\Xiangping\\498\\Segm_Ident_Vertebrae_CNN_kmeans_knn-master\\')
from config.paths import base_dataset_dir

path_to_original_images_ = "dataset-verse19training/images"
path_to_original_labels_ = "dataset-verse19training/labels"

path_to_validation_images_ = "dataset-verse19validation/images"
path_to_validation_labels_ = "dataset-verse19validation/labels"


def run(path_to_original_images, path_to_original_labels,
        path_to_validation_images, path_to_validation_labels):
    path_to_original_images = os.path.join(base_dataset_dir, path_to_original_images)
    path_to_original_labels = os.path.join(base_dataset_dir, path_to_original_labels)
    path_to_validation_images = os.path.join(base_dataset_dir, path_to_validation_images)
    path_to_validation_labels = os.path.join(base_dataset_dir, path_to_validation_labels)

    original_length = len(os.listdir(path_to_original_images))
    original_length_l = len(os.listdir(path_to_original_labels))

    os.makedirs(path_to_validation_images, exist_ok=True)
    os.makedirs(path_to_validation_labels, exist_ok=True)

    validation_labels = ['sub-verse010_seg-vert_msk.nii.gz', 
                         'sub-verse011_seg-vert_msk.nii.gz', 
                         'sub-verse013_seg-vert_msk.nii.gz', 
                         'sub-verse016_seg-vert_msk.nii.gz', 
                         'sub-verse018_seg-vert_msk.nii.gz', 
                         'sub-verse022_seg-vert_msk.nii.gz', 
                         'sub-verse023_seg-vert_msk.nii.gz', 
                         'sub-verse024_seg-vert_msk.nii.gz', 
                         'sub-verse026_seg-vert_msk.nii.gz', 
                         'sub-verse030_seg-vert_msk.nii.gz', 
                         'sub-verse041_seg-vert_msk.nii.gz', 
                         'sub-verse047_seg-vert_msk.nii.gz', 
                         'sub-verse058_seg-vert_msk.nii.gz', 
                         'sub-verse067_seg-vert_msk.nii.gz', 
                         'sub-verse071_seg-vert_msk.nii.gz', 
                         'sub-verse073_seg-vert_msk.nii.gz', 
                         'sub-verse078_seg-vert_msk.nii.gz', 
                         'sub-verse080_seg-vert_msk.nii.gz', 
                         'sub-verse093_seg-vert_msk.nii.gz', 
                         'sub-verse095_seg-vert_msk.nii.gz', 
                         'sub-verse116_seg-vert_msk.nii.gz', 
                         'sub-verse124_seg-vert_msk.nii.gz', 
                         'sub-verse125_seg-vert_msk.nii.gz', 
                         'sub-verse150_seg-vert_msk.nii.gz', 
                         'sub-verse153_seg-vert_msk.nii.gz', 
                         'sub-verse205_seg-vert_msk.nii.gz', 
                         'sub-verse221_seg-vert_msk.nii.gz', 
                         'sub-verse225_seg-vert_msk.nii.gz', 
                         'sub-verse230_seg-vert_msk.nii.gz', 
                         'sub-verse242_seg-vert_msk.nii.gz', 
                         'sub-verse252_seg-vert_msk.nii.gz', 
                         'sub-verse264_seg-vert_msk.nii.gz', 
                         'sub-verse269_seg-vert_msk.nii.gz', 
                         'sub-verse276_seg-vert_msk.nii.gz', 
                         'sub-verse400_split-verse090_seg-vert_msk.nii.gz', 
                         'sub-verse400_split-verse155_seg-vert_msk.nii.gz', 
                         'sub-verse404_split-verse209_seg-vert_msk.nii.gz', 
                         'sub-verse404_split-verse256_seg-vert_msk.nii.gz', 
                         'sub-verse412_split-verse235_seg-vert_msk.nii.gz', 
                         'sub-verse412_split-verse290_seg-vert_msk.nii.gz']
    validation_images = [vl.replace('_seg-vert_msk', '_ct') for vl in validation_labels]

    assert len(validation_images) == len(validation_labels), "Mismatch in sizes between validation images and labels"
    
    # train
    train_images = []
    for image in os.listdir(path_to_original_images):
        if image.endswith('.gz'):
            train_images.append(image)
    train_labels = []
    for label in os.listdir(path_to_original_labels):
        if label.endswith('.gz'):
            train_labels.append(label)

    # validation
    val_images = []
    for image in os.listdir(path_to_validation_images):
        if image.endswith('.gz'):
            val_images.append(image)
    val_labels = []
    for label in os.listdir(path_to_validation_labels):
        if label.endswith('.gz'):
            val_labels.append(label)

    val_images.sort()
    val_labels.sort()

    cnt_val = 0

    for train_image, train_label, val_image, val_label in zip(train_images, train_labels, val_images, val_labels):
        is_val_i = val_image in validation_images
        is_val_l = val_label in validation_labels
        
        assert is_val_i == is_val_l, "Mismatch between validation image and label! {} and {}".format(trainval_image, trainval_label)

        if is_val_i:
            cnt_val += 1
            source_image = os.path.join(path_to_original_images, train_image)
            destination_image = os.path.join(path_to_validation_images, val_image)
            shutil.move(source_image, destination_image)

            source_label = os.path.join(path_to_original_labels, train_label)
            destination_label = os.path.join(path_to_validation_labels, val_label)
            shutil.move(source_label, destination_label)

    print("Counter Val  = ", cnt_val)
    print("Validation I = ", len(validation_images))
    print("Validation L = ", len(validation_labels))
    assert cnt_val == len(validation_images) == len(validation_labels), "Mismatch in size between cnt_val and validation set"

    train_len_i = len(os.listdir(path_to_original_images))
    val_len_i = len(os.listdir(path_to_validation_images))

    assert train_len_i + val_len_i == original_length, "Mismatch after train-val split"

    train_len_l = len(os.listdir(path_to_original_labels))
    val_len_l = len(os.listdir(path_to_validation_labels))

    assert train_len_l + val_len_l == original_length_l, "Mismatch in labels after train-val split"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for performing trainval/test splitting")
    parser.add_argument(
        "-oti",
        "--original-training-images",
        default=path_to_original_images_, type=str,
        help="Specify the path where there are the original images"
    )
    parser.add_argument(
        "-otl",
        "--original-training-labels",
        default=path_to_original_labels_, type=str,
        help="Specify the path where there are the original labels"
    )
    parser.add_argument(
        "-ovi",
        "--original-validation-images",
        default=path_to_validation_images_, type=str,
        help="Specify the path where to put the original validation images"
    )
    parser.add_argument(
        "-ovl",
        "--original-validation-labels",
        default=path_to_validation_labels_, type=str,
        help="Specify the path where to put the original validation labels"
    )
    args = parser.parse_args()
    run(
        args.original_training_images,
        args.original_training_labels,
        args.original_validation_images,
        args.original_validation_labels
    )
