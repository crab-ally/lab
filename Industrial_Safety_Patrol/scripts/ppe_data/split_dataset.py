#!/usr/bin/env python3

import os
import random
import shutil


################################################
# Dataset path
################################################

DATASET="/workspace/datasets/ppe_dataset"


IMAGE_SRC=os.path.join(
    DATASET,
    "images"
)

LABEL_SRC=os.path.join(
    DATASET,
    "labels"
)


################################################
# Output path
################################################

TRAIN_IMAGE=os.path.join(
    DATASET,
    "images/train"
)

VAL_IMAGE=os.path.join(
    DATASET,
    "images/val"
)


TRAIN_LABEL=os.path.join(
    DATASET,
    "labels/train"
)

VAL_LABEL=os.path.join(
    DATASET,
    "labels/val"
)



for path in [

    TRAIN_IMAGE,
    VAL_IMAGE,
    TRAIN_LABEL,
    VAL_LABEL

]:

    os.makedirs(
        path,
        exist_ok=True
    )



################################################
# Split ratio
################################################

TRAIN_RATIO=0.8



################################################
# Image list
################################################

images=[]


for file in os.listdir(IMAGE_SRC):

    if file.endswith(".jpg"):

        images.append(file)



print(
    f"Total images : {len(images)}"
)



################################################
# Shuffle
################################################

random.seed(42)

random.shuffle(
    images
)



train_count=int(
    len(images)
    *
    TRAIN_RATIO
)



train_files=images[:train_count]

val_files=images[train_count:]



print(
    f"Train : {len(train_files)}"
)

print(
    f"Val   : {len(val_files)}"
)



################################################
# Copy
################################################

def copy_files(files, image_dst, label_dst):


    for image in files:


        ################################################
        # image copy
        ################################################

        shutil.copy(

            os.path.join(
                IMAGE_SRC,
                image
            ),

            os.path.join(
                image_dst,
                image
            )

        )


        ################################################
        # label copy
        ################################################

        label=image.replace(
            ".jpg",
            ".txt"
        )


        label_path=os.path.join(
            LABEL_SRC,
            label
        )


        if os.path.exists(label_path):

            shutil.copy(

                label_path,

                os.path.join(
                    label_dst,
                    label
                )

            )

        else:

            print(
                "Missing label:",
                label
            )



################################################
# Execute
################################################

copy_files(

    train_files,

    TRAIN_IMAGE,

    TRAIN_LABEL

)


copy_files(

    val_files,

    VAL_IMAGE,

    VAL_LABEL

)



print(
    "Dataset split complete"
)