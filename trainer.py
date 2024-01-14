from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import hypertune

from sklearn.metrics import classification_report, confusion_matrix

import os
import numpy as np

from DataHandler import download_data_to_local_directory, upload_data_to_bucket

from tensorflow.python.client import device_lib
import argparse
import shutil # to create zip files
from datetime import datetime

# build BASE MODEL (InceptionV3) existing model trained on dataset called ‘Imagenet’.
# will use above for transfer learning, but will train new layers with own dataset
# include_top=False below is to remove output layers(classes) and replace with our new layers
# input_tensor (input size of neural network), input layer shape 229, 229, 3 (RGB channels)
def build_model(nbr_classes):
    base_model = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(229, 229, 3)))

    head_model = base_model.output
    head_model = Flatten()(head_model) # flattens (output layer) what comes before new layer
    head_model = Dense(512, activation='relu')(head_model)
    # create fully connected new Dense layer (choose 512 option number of neurons
    head_model = Dropout(0.5)(head_model)
    # hyperparameter 'Dropout' able to be changed each time, to improve deep learning model performance
    head_model = Dense(nbr_classes, activation="softmax")(head_model)
    # final CNN dense layer number of classes as parameter (number of classes as number of neurons given)
    # activation function - takes some input and maps to certain output ('softmax' for classes independent of each other)

    model = Model(inputs=base_model.input, outputs=head_model)
    # final model use base model as input
    # head model as output (adapted base model output with the new layers, final tensor of our CNN)

    # go through base model layers parameters and make them NOT trainable, only to train NEW layers (head model)
    for layer in base_model.layers:
        layer.trainable = False

    return model

# Create data pipelines to feed data to model during training and validation and also evaluation after training
def build_data_pipelines(batch_size, train_data_path, val_data_path, eval_data_path):
    # give set of transformations that will be applied to images
    # called augmentor as term used by ML/DL engineers, function takes original images, applies transformations on them
    # generator created at end that will use same augmentor to generate new images based in transformations defined here
    # goal is to generate new data for training with transformation parameters...
    # to teach model to see as many different scenarios as possible
    train_augmentor = ImageDataGenerator(
        rescale=1. / 255,  # rescale is only transformation that is NOT optional
        # divide pixel values by 255 (as InceptionV3 takes values between 0 and 1 on ImageNet dataset)
        # helps 'regularisation' which is to avoid over-fitting
        # essential transformation for generating new images from existing dataset
        # OPTIONAL TRANSFORMATIONS - THESE VALUES MAY BE ADAPTED TO OPTIMISE TRAINING:
        rotation_range=25, # images rotated between 0 - 25 degrees range
        zoom_range=0.15, # images zoom between 0 - 0.15 degrees range
        width_shift_range=0.2, # shift images with 20% value of width (horizontal axis)
        height_shift_range=0.2, # shift images with 20% value of height (height axis)
        shear_range=0.15, # move pixels in image in certain direction
        horizontal_flip=True, # flip images from left to right (horizontal like in mirror)
        fill_mode="nearest" # method to fill pixels with no value use certain function called 'nearest',others eg'cubic'
    )

    # validation and evaluation augmentor will be different
    # don't want to introduce transformations as goal is to see how model will perform on real data
    # real data will not have optional transformations added
    # purpose for transformations is to train model to see as many different scenarios as possible
    # only transformation applied is non-optional rescale
    val_augmentor = ImageDataGenerator(
        rescale=1. / 255
    )

    # create generators from same augmentors as above, to generate new images based in transformations defined above
    # flow_from_directory is function that takes path to training images as an input
    # applies transformations on images using train augmentor and will generate new images as output
    train_generator = train_augmentor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        # define what kind of classes, categorical means each being different (independent of each other)
        # could also define as mixed with multiple classes in image
        target_size=(229, 229), # size of images after transformation (same size as input for neural network)
        color_mode="rgb",
        shuffle=True, # shuffle data every new batch for as much randomness as possible during training
        # so neural network sees cases in non-monotonic way
        batch_size=batch_size
    )


    val_generator = val_augmentor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(229, 229),
        color_mode="rgb",
        shuffle=False, # do NOT shuffle data during validation/evaluation phase, randomness not needed
        batch_size=batch_size
    )

    eval_generator = val_augmentor.flow_from_directory(
        eval_data_path,
        class_mode="categorical",
        target_size=(229, 229),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )

    return train_generator, val_generator, eval_generator

# GET NUMBER OF IMAGES IN DIRECTORY
def get_number_of_imgs_inside_folder(directory):
    totalcount = 0
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in [".png", ".jpg", ".jpeg"]:
                totalcount = totalcount + 1

    return totalcount

# DEFINE TRAIN FUNCTION
def train(path_to_data, batch_size, epochs, learning_rate, models_bucket_name):
    path_train_data = os.path.join(path_to_data, 'training')
    path_val_data = os.path.join(path_to_data, 'validation')
    path_eval_data = os.path.join(path_to_data, 'evaluation')

    total_train_imgs = get_number_of_imgs_inside_folder(path_train_data)
    total_val_imgs = get_number_of_imgs_inside_folder(path_val_data)
    total_eval_imgs = get_number_of_imgs_inside_folder(path_eval_data)

    print(total_train_imgs, total_val_imgs, total_eval_imgs)

    train_generator, val_generator, eval_generator = build_data_pipelines(
        batch_size=batch_size,
        train_data_path=path_train_data,
        val_data_path=path_val_data,
        eval_data_path=path_eval_data
    )

    classes_dict = train_generator.class_indices
    model = build_model(nbr_classes=len(classes_dict.keys())) # automate from hard 11 to len(classes_dict.keys()) param above

    optimizer = Adam(lr=learning_rate)  # 1e-5

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    path_to_save_model = './tmp'
    if not os.path.isdir(path_to_save_model):
        os.makedirs(path_to_save_model)

    ckpt_saver = ModelCheckpoint(
        path_to_save_model,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=total_train_imgs // batch_size, #tells model that one epoch (run of dataset) has happened once and to stop
        validation_data=val_generator,
        validation_steps=total_val_imgs // batch_size,
        epochs=epochs,
        callbacks=[early_stopping, ckpt_saver]
    )

    print("[INFO] Evaluation phase...")

    predictions = model.predict_generator(eval_generator)
    predictions_idxs = np.argmax(predictions, axis=1)

    my_classification_report = classification_report(eval_generator.classes, predictions_idxs,
                                                     target_names=eval_generator.class_indices.keys())

    my_confusion_matrix = confusion_matrix(eval_generator.classes, predictions_idxs)

    print("[INFO] Classification report : ")
    print(my_classification_report)

    print("[INFO] Confusion matrix : ")
    print(my_confusion_matrix)

    print("Starting evaluation using model.evaluate_generator")
    scores = model.evaluate_generator(eval_generator)
    print("Done evaluating!")
    loss = scores[0]
    print(f"loss for hypertune = {loss}")

    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    zipped_folder_name = f'trained_model_{now}_loss_{loss}'

    shutil.make_archive(zipped_folder_name, 'zip', '/usr/src/app/tmp')  # only runs on cloud, not locally!

    path_zipped_folder = '/usr/src/app/' + zipped_folder_name + '.zip'
    upload_data_to_bucket(models_bucket_name, path_zipped_folder, zipped_folder_name)

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='loss',
                                            metric_value=loss, global_step=epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bucket_name", type=str, help="Bucket name on google cloud storage",
                        default="foods-data-bucket")

    parser.add_argument("--models_bucket_name", type=str,
                        help="Bucket name on google cloud storage for saving trained models",
                        default="trained_models_foods_classification")

    parser.add_argument("--batch_size", type=int, help="Batch size used by the deep learning model",
                        default=2)

    parser.add_argument("--learning_rate", type=float, help="Batch size used by the deep learning model",
                        default=1e-5)

    args = parser.parse_args()


    print("Downloading of data started ...")
    download_data_to_local_directory(args.bucket_name, "./data")
    print("Download finished!")

    path_to_data = './data'
    train(path_to_data, args.batch_size, 20, args.learning_rate, args.models_bucket_name)
    # FOR TESTING  train(path_to_data, 2, 2)


