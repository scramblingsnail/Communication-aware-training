# Randomly extract 100 classes from ImageNet
import os
import json
import scipy.io as sio
import shutil, os
import random

from typing import Tuple


def read_image_net_word_idx(meta_data_path: str) -> Tuple[dict, dict]:
    meta_data = sio.loadmat(meta_data_path)
    idx_to_word = dict()
    word_to_idx = dict()

    words_info = meta_data["synsets"]
    for each_info in words_info:
        label = int(each_info[0][0][0][0])
        word = each_info[0][1][0]
        idx_to_word[label] = word
        word_to_idx[word] = label
        # print(label, word)
    return idx_to_word, word_to_idx


def construct_image_net_val_dir(
    val_labels_path: str,
    val_images_dir: str,
    target_val_dir: str,
    label_to_word: dict,
):
    r"""
    Categorize the validation data.

    Args:
        val_labels_path (str): The ground truths of validation data
        val_images_dir (str): The directory that stores the validation images.
        target_val_dir (str): The directory that stores the categorized validation images.
        label_to_word (dict): key: label (start from 1); val: word_name (dir_name)

    Returns:

    """
    # read labels; number labels start from 1.
    with open(val_labels_path, "r") as label_f:
        label_lines = label_f.readlines()
    # ground truth of val images.
    labels = [int(line.strip()) for line in label_lines if len(line) > 0]
    dir_names = [label_to_word[l] for l in labels]

    if not os.path.exists(target_val_dir):
        os.makedirs(target_val_dir)

    src_img_prefix = "ILSVRC2012_val_"
    val_img_names = []
    # record val_img_f_name
    for val_idx, label in enumerate(labels):
        # in src name, idx start from 1.
        val_idx_str = str(val_idx + 1).rjust(8, "0")
        img_name = f"{src_img_prefix}{val_idx_str}.JPEG"
        val_img_names.append(img_name)

    for val_idx, img_f_name in enumerate(val_img_names):
        src_path = f"{val_images_dir}/{img_f_name}"
        word_name = dir_names[val_idx]
        target_dir = f"{target_val_dir}/{word_name}"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        target_path = f"{target_dir}/{img_f_name}"
        shutil.copy(src_path, target_path)
        print("Src: ", src_path)
        print("Target: ", target_path)


def image_net_sub_set(
    subset_dir: str,
    idx_to_word: dict,
    src_train_data_dir: str,
    src_val_data_dir: str,
    categories_num: int,
    exclude_classes: list = None,
    specified_classes: list = None,
):
    r"""
    Randomly select a sub-dataset.

    Args:
        subset_dir (str): The directory that stores the sub-dataset.
        idx_to_word (dict): key: label (start from 1); val: word_name (dir_name)
        src_train_data_dir (str): The directory that stores the source train dataset.
        src_val_data_dir (str): The directory that stores the source validation dataset.
        categories_num (int): The categories num in the sub-dataset.
        exclude_classes (list):
        specified_classes (list):

    Returns:

    """
    if not os.path.exists(subset_dir):
        os.makedirs(subset_dir)

    if categories_num > len(idx_to_word.keys()):
        raise ValueError("The categories in the sub-dataset cannot be more than that of the source dataset.")

    src_dir_names = os.listdir(src_train_data_dir)
    if exclude_classes:
        src_dir_names = [n for n in src_dir_names if n not in exclude_classes]

    valid_classes = []
    for each_class in src_dir_names:
        train_class_path = f"{src_train_data_dir}/{each_class}"
        class_images = os.listdir(train_class_path)
        if len(class_images) > 800:
            valid_classes.append(each_class)

    print(len(valid_classes))
    select_dir_names = random.sample(valid_classes, categories_num)

    subset_train_dir = f"{subset_dir}/train"
    subset_val_dir = f"{subset_dir}/val"

    if specified_classes:
        select_dir_names = specified_classes
        print("Using specified classes")

    for dir_name in select_dir_names:
        each_src_train_dir = f"{src_train_data_dir}/{dir_name}"
        each_src_val_dir = f"{src_val_data_dir}/{dir_name}"
        each_target_train_dir = f"{subset_train_dir}/{dir_name}"
        each_target_val_dir = f"{subset_val_dir}/{dir_name}"
        print("Src train dir: ", each_src_train_dir)
        print("Target train dir: ", each_target_train_dir)
        print("Src val dir: ", each_src_val_dir)
        print("Target val dir: ", each_target_val_dir)
        if not os.path.exists(each_src_train_dir) or not os.path.exists(each_src_val_dir):
            raise ValueError("Does not exist.")
        shutil.copytree(src=each_src_train_dir, dst=each_target_train_dir)
        shutil.copytree(src=each_src_val_dir, dst=each_target_val_dir)
    return select_dir_names


def generate_sub_train_dataset(
    src_data_dir: str,
    train_data_num: int,
    repeat_times: int,
):
    target_dir = f"{src_data_dir}-{train_data_num}"
    src_train_data_dir = f"{src_data_dir}/train"
    src_val_data_dir = f"{src_data_dir}/val"

    src_train_names = os.listdir(src_train_data_dir)
    # extract train_data_num to the target_dir
    for class_name in src_train_names:
        target_class_data_dir = f"{target_dir}/train/{class_name}"
        if not os.path.exists(target_class_data_dir):
            os.makedirs(target_class_data_dir)

        src_class_data_dir = f"{src_train_data_dir}/{class_name}"
        image_names = os.listdir(src_class_data_dir)
        # print(len(image_names))
        select_images = random.sample(image_names, train_data_num)
        for name in select_images:
            src_image_path = f"{src_class_data_dir}/{name}"
            for repeat_idx in range(repeat_times):
                target_image_path = f"{target_class_data_dir}/No{repeat_idx}_{name}"
                print(f"Copying from {src_image_path} to {target_image_path}")
                shutil.copyfile(src_image_path, target_image_path)

    # copy val data
    target_val_data_dir = f"{target_dir}/val"
    print(f"Copying from {src_val_data_dir} to {target_val_data_dir}")
    shutil.copytree(src_val_data_dir, target_val_data_dir)

    # copy mean and std
    src_mean_path = f"{src_data_dir}/train_mean.pt"
    src_std_path = f"{src_data_dir}/train_std.pt"
    target_mean_path = f"{target_dir}/train_mean.pt"
    target_std_path = f"{target_dir}/train_std.pt"
    shutil.copyfile(src_mean_path, target_mean_path)
    shutil.copyfile(src_std_path, target_std_path)


def record_class_names(version_idx: int):
    subset_dir = f"/root/autodl-tmp/imagenet-100-v{version_idx}"
    subset_train_dir = f"{subset_dir}/train"
    class_names = os.listdir(subset_train_dir)

    class_names_dir = r"/root/zhisan/DistHybridQuantize/checkpoints/imagenet_data_names"
    if not os.path.exists(class_names_dir):
        os.makedirs(class_names_dir)
    class_names_path = fr"{class_names_dir}/names-v{version_idx}.txt"
    with open(class_names_path, "w") as class_f:
        for name in class_names:
            class_f.write(f"{name}\n")

def load_class_names(version_idx: int):
    class_names_dir = r"/root/zhisan/DistHybridQuantize/checkpoints/imagenet_data_names"
    class_names_path = fr"{class_names_dir}/names-v{version_idx}.txt"
    with open(class_names_path, "r") as name_f:
        names = name_f.readlines()

    class_names = [n.strip() for n in names]
    return class_names


if __name__ == "__main__":
    val_ground_truth_path = "/root/autodl-tmp/imagenet/devkit/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
    val_img_dir = "/root/autodl-tmp/imagenet/raw_val"
    target_img_dir = "/root/autodl-tmp/imagenet/val"
    num_to_word, _ = read_image_net_word_idx("/root/autodl-tmp/imagenet/devkit/ILSVRC2012_devkit_t12/data/meta.mat")

    # Extract val data
    construct_image_net_val_dir(
        val_labels_path=val_ground_truth_path,
        val_images_dir=val_img_dir,
        target_val_dir=target_img_dir,
        label_to_word=num_to_word,
    )


    # Randomly extract ImageNet-100
    # used_classes = []
    # for v_idx in range(5, 6):
    #     name_list = load_class_names(version_idx=v_idx)
    #     each_used_classes = image_net_sub_set(
    #         subset_dir=f"/root/autodl-tmp/imagenet-100-v{v_idx}",
    #         idx_to_word=num_to_word,
    #         src_train_data_dir="/root/autodl-tmp/imagenet/train",
    #         src_val_data_dir="/root/autodl-tmp/imagenet/val",
    #         categories_num=100,
    #         exclude_classes=used_classes,
    #         specified_classes=name_list,
    #     )
    #     used_classes.extend(each_used_classes)


    # Randomly extract train dataset with different size 100 200 400 800
    # train_num_list = [100, 200, 400, 800]
    # for v_idx in range(1, 6):
    #     print(v_idx)
    #     src_dataset_dir = f"/root/autodl-tmp/imagenet-100-v{v_idx}"
    #     for data_num in train_num_list:
    #         generate_sub_train_dataset(
    #             src_data_dir=src_dataset_dir,
    #             train_data_num=data_num,
    #             repeat_times=train_num_list[-1] // data_num,
    #         )


    # for v_idx in range(1, 6):
    #     record_class_names(version_idx=v_idx)
