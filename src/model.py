import os
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from keras.models import Sequential
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.optimizers import Adam
import matplotlib.pyplot as plt


CURRENT_DIRECTORY = os.curdir


#def load_data():
def loading_image_data():

    #lOADING DATA FROM THE CURRENT DIRECTORY AND RETURN IMAGES AS PER GOOD OR BAD NAILSGUNS

    good_nails_path = f'{CURRENT_DIRECTORY}/data/nailgun/good'
    bad_nails_path = f'{CURRENT_DIRECTORY}/data/nailgun/bad'

    good_nails = map((lambda x: f'{good_nails_path}/{x}'), os.listdir(good_nails_path))
    bad_nails = map((lambda x: f'{bad_nails_path}/{x}'), os.listdir(bad_nails_path))
    files = list(good_nails) + list(bad_nails)
    all_nailguns = []
    for image in files:
        if '.jpeg' in image:
            all_nailguns.append(image)

    dataframe = pd.DataFrame(data=all_nailguns, columns=['image'])

    dataframe['label'] = dataframe['image'].map(lambda image_name: 'good' if 'good' in image_name else 'bad')

    return dataframe


def train_test_split(dataframe):
    #Train and Test data split_dataset


    data = dataframe['image']
    label = dataframe['label']
    split_dataset = ShuffleSplit(n_splits=1, test_size=0.33, random_state=0)
    for train_index, test_index in split_dataset.split(data, label):
        training_data, test_data = data[train_index], data[test_index]
        training_label, test_label = label[train_index], label[test_index]
    training_data = pd.concat([training_data, training_label], axis=1)
    testing_data = pd.concat([test_data, test_label], axis=1)
    return training_data, testing_data


def image_preprocessing(training_data, testing_data, class_mode):
    # image processing and data augmentation for preparation before fitting into classification model.
    image_size = [224, 224]
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True,
                                        vertical_flip=True, zoom_range=0.2, rotation_range=40,
                                        fill_mode='constant')

    train_generator = data_generator.flow_from_dataframe(
        training_data,
        x_col='image',
        y_col='label',
        batch_size=5,
        target_size=image_size,
        drop_duplicates=True,
        class_mode=class_mode)

    test_generator = data_generator.flow_from_dataframe(
        testing_data,
        x_col='image',
        y_col='label',
        batch_size=5,
        target_size=image_size,
        drop_duplicates=True,
        class_mode=class_mode)
    return train_generator, test_generator


# def baseline_cnn_model():
#
#     baseline_model = Sequential()
#     baseline_model.add(Conv2D(32, (2, 2), input_shape=(224, 224, 3)))
#     baseline_model.add(Activation('elu'))
#     baseline_model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     baseline_model.add(Conv2D(32, (2, 2)))
#     baseline_model.add(Activation('elu'))
#     baseline_model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     baseline_model.add(Conv2D(64, (2, 2)))
#     baseline_model.add(Activation('elu'))
#     baseline_model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     baseline_model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
#     baseline_model.add(Dense(64))
#     baseline_model.add(Activation('elu'))
#     baseline_model.add(Dropout(0.5))
#     baseline_model.add(Dense(1))
#     baseline_model.add(Activation('sigmoid'))
#
#     # Compile the model
#
#     baseline_model.compile(optimizer=RMSprop(lr=0.001, loss='binary_crossentropy', metrics=['accuracy'])
#
#     return  baseline_model

def get_baseline_model():




    baseline_model = Sequential()
    baseline_model.add(Conv2D(32, (2, 2), input_shape=(224, 224, 3)))
    baseline_model.add(Activation('relu'))
    baseline_model.add(MaxPooling2D(pool_size=(2, 2)))

    baseline_model.add(Conv2D(32, (2, 2)))
    baseline_model.add(Activation('relu'))
    baseline_model.add(MaxPooling2D(pool_size=(2, 2)))

    baseline_model.add(Conv2D(64, (2, 2)))
    baseline_model.add(Activation('relu'))
    baseline_model.add(MaxPooling2D(pool_size=(2, 2)))

    baseline_model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    baseline_model.add(Dense(64))
    baseline_model.add(Activation('relu'))
    baseline_model.add(Dropout(0.5))
    baseline_model.add(Dense(1))
    baseline_model.add(Activation('sigmoid'))

    # Compile the model
    baseline_model.compile(optimizer=RMSprop(lr=0.001),loss='binary_crossentropy', metrics=['accuracy'])
    return baseline_model


def vgg16_cnn_model():
    """
    Network architecture definition for VGG-16 model

    :return: compiled VGG-16 model
    """
    vgg_model = Sequential()
    vgg_model.add(VGG16(weights='imagenet', include_top=False, pooling='avg'))
    #vgg_model.add(Dense(units=2, activation='softmax'))
    vgg_model.add(Dense(units=2, activation='softmax'))
    vgg_model.layers[0].trainable = False

    # Compile the model

    #vgg_model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])
    vgg_model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return vgg_model


def training_baseline_model():

    checkpoint_filepath = f'{CURRENT_DIRECTORY}/model/baseline-cnn-model.hdf5'

    dataframe = loading_image_data()
    training_data, test_data = train_test_split(dataframe)
    train_generator, test_generator = image_preprocessing(training_data, test_data, class_mode='binary')

    # Get baseline model
    baseline_model = get_baseline_model()

    # Configuration to checkpoint the model
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc',save_best_only=True, mode='max')

    # Fit the baseline model
    baseline_model.fit_generator(
            train_generator,
            steps_per_epoch=3,
            validation_data=test_generator,
            validation_steps=1,
            verbose=1,
            epochs=15,
            callbacks=[checkpoint])




def training_vgg16_model():

    checkpoint_filepath = f'{CURRENT_DIRECTORY}/model/vgg16-classifier-model.hdf5'

    dataframe = loading_image_data()
    training_data, test_data = train_test_split(dataframe)
    train_generator, test_generator = image_preprocessing(training_data, test_data, class_mode='categorical')

    # Get VGG-16 model
    vgg_model = vgg16_cnn_model()

    # Configuration to checkpoint the model
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc',  save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Fit the VGG-16 model
    vgg_model.fit_generator(
            train_generator,
            steps_per_epoch=3,
            validation_data=test_generator,
            validation_steps=1,
            verbose=1,
            epochs=15,
            callbacks=callbacks_list)






if __name__ == '__main__':

    training_baseline_model()
    training_vgg16_model()
