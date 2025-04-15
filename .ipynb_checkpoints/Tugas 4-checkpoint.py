# =====================================================================================================
# Membangun sebuah model Neural Network untuk klasifikasi dataset Horse or Human dalam binary classes.
#
# Input layer harus menerima 150x150 dengan 3 bytes warna sebagai input shapenya.
# Jangan menggunakan lambda layers dalam model.
#
# Dataset yang digunakan dibuat oleh Laurence Moroney (laurencemoroney.com).
#
# Standar yang harus dicapai untuk accuracy dan validation_accuracy > 83%
# =====================================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


def solution_05():
    data_url_1 = 'https://github.com/dicodingacademy/assets/releases/download/release-horse-or-human/horse-or-human.zip'
    urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')
    local_file = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/horse-or-human')

    data_url_2 = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/validation-horse-or-human.zip'
    urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
    local_file = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/validation-horse-or-human')
    zip_ref.close()

    TRAINING_DIR = 'data/horse-or-human'
    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    train_generator= ImageDataGenerator(rescale=1./255)

    # Compile the Model
    model.compile(loss="binary_crossentropy",
                optimizer=RMSprop(lr=0.001),
                metrics=["accuracy"])

    # Training the Model
    history = model.fit(
        train_generator,
        steps_per_epoch=8, 
        epochs=15,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=8
    )

    model=tf.keras.models.Sequential([
                # The first convolution
                # Input image has 3 bytes color
                tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(300, 300, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                # The second convolution
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                # The third convolution
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                # The fourth convolution
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                # The fifth convolution
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                # Flatten the results to feed in Deep Neural Network
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(1, activation='sigmoid') #DO NOT CHANGE THIS LINE!
        ])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_05()
    model.save("model_05.h5")
