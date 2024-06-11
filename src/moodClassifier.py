# people_mood_classifier_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall

def limit_gpu_memory_growth():
    gpu_available = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpu_available:
        tf.config.experimental.set_memory_growth(gpu, True)

def clean_data(data_dir):
    image_available = ['jpeg', 'jpg', 'bmp', 'png']
    for imageClass in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, imageClass)):
            imagePath = os.path.join(data_dir, imageClass, image)
            try:
                imageValid = cv2.imread(imagePath)
                checkExt = imghdr.what(imagePath)
                if checkExt not in image_available:
                    print(f"The extension of {imagePath} does not exist")
                    os.remove(imagePath)
                else:
                    image_size_kb = os.path.getsize(imagePath) / 1024
                    if image_size_kb < 2:
                        print(f"Removing {imagePath} (size: {image_size_kb:.2f} KB)")
                        os.remove(imagePath)
            except Exception as e:
                print(f"The image {imagePath} has an extension issue or cannot be read.")

def load_data(data_dir):
    data = tf.keras.utils.image_dataset_from_directory(data_dir)
    data = data.map(lambda x, y: (x / 255, y))
    return data

def split_data(data):
    train_size = int(len(data) * 0.75)
    val_size = int(len(data) * 0.23)
    test_size = int(len(data) * 0.05)
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)
    return train, val, test

def build_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model
