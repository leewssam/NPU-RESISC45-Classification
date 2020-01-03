import numpy as np
from keras.applications import VGG16, ResNet50
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img
import keras

train_dir = './data/train'
validation_dir = './data/test'

vgg_conv = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))


def get_features():
    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 32
    n_train = 6600
    c = 12
    image_size=224
    
    train_features = np.zeros(shape=(n_train, 7, 7, 512))
    train_labels = np.zeros(shape=(n_train, c))
    
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    i = 0
    for inputs_batch, labels_batch in train_generator:
        features_batch = vgg_conv.predict(inputs_batch)
        train_features[i * batch_size : (i + 1) * batch_size] = features_batch
        train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= nTrain:
            break
   
    train_features = np.reshape(train_features, (n_train, 7 * 7 * 512))

    n_val = 1800
    validation_features = np.zeros(shape=(n_val, 7, 7, 512))
    validation_labels = np.zeros(shape=(n_val, c))

    validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    i = 0
    for inputs_batch, labels_batch in validation_generator:
        features_batch = vgg_conv.predict(inputs_batch)
        validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
        validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= nVal:
            break

    validation_features = np.reshape(validation_features, (n_val, 7 * 7 * 512))

    return train_features, train_labels, validation_features, validation_labels