import numpy as np
import cv2
import random
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


letter_dir = 'base_letters/'
img_size = 32
batch = 32
epoch = 50
charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
num_classes = len(charset)
random.seed()
sample_count = batch * 4000
input_shape = (img_size, img_size, 1)
wiggle = 5


def add_gaussian_noise(image):
    gauss = np.random.normal(loc=0, scale=20, size=image.shape)
    gauss = gauss.reshape(image.shape)
    noisy = image + gauss
    return noisy


def crop_image_random(image):
    size = image.shape
    x = round(size[0] / 2 - img_size / 2)
    y = round(size[1] / 2 - img_size / 2)
    x += random.randint(-wiggle, wiggle)  # add lateral variance
    y += random.randint(0, wiggle*2)  # add vertical variance
    return image[y:y + img_size, x:x + img_size]


def random_brightness(image):
    image += random.randint(-175, 75)
    low_vals = image < 0
    image[low_vals] = 0
    high_vals = image > 255
    image[high_vals] = 255
    return image


def random_size(image):
    factor = random.uniform(0.5, 1.5)
    size = round(image.shape[0] * factor)
    cv2.resize(image, dst=image, dsize=(size, size), fx=factor, fy=factor)
    return image


def randomize_image(image):
    image = add_gaussian_noise(image)
    image = random_brightness(image)
    image = random_size(image)
    image = crop_image_random(image)
    return image


def generate_samples(count):
    # this can be accelerated by threading/multiprocessing
    files = os.listdir(letter_dir)
    secure_random = random.SystemRandom()
    results = []
    for i in range(count):
        file = secure_random.choice(files)
        img = cv2.imread(letter_dir + file)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = randomize_image(img)
        label = int(file.split('_')[1])
        results.append((img, label))
        if i % 500 == 0:
            print('SAMPLES', i, 'OF', count)
    return results


def instantiate_model():
    print('compiling model')
    model = Sequential()

    model.add(Convolution2D(filters=32,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            input_shape=input_shape))

    model.add(Convolution2D(filters=32,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,
                    activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(num_classes,
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())
    return model


def prepare_datasets(samples):
    random.shuffle(samples)
    training_set = samples[:round(sample_count / 10 * 9)]
    test_set = samples[round(sample_count / 10 * 9):]

    training_data = [i[0] for i in training_set]
    training_data = np.asarray(training_data).astype('float32')
    training_data = np.expand_dims(training_data, axis=3)
    training_data /= 255
    training_labels = [i[1] for i in training_set]
    training_labels = np.asarray(training_labels)
    training_labels = np_utils.to_categorical(training_labels, num_classes)

    print('TRAINING DATA SHAPE', training_data.shape)
    print('TRAINING LABELS SHAPE', training_labels.shape)

    test_data = [i[0] for i in test_set]
    test_data = np.asarray(test_data).astype('float32')
    test_data = np.expand_dims(test_data, axis=3)
    test_data /= 255
    test_labels = [i[1] for i in test_set]
    test_labels = np.asarray(test_labels)
    test_labels = np_utils.to_categorical(test_labels, num_classes)

    print('TEST DATA SHAPE', test_data.shape)
    print('TEST LABELS SHAPE', test_labels.shape)
    return training_data, training_labels, test_data, test_labels


if __name__ == '__main__':
    samples = generate_samples(count=sample_count)
    training_data, training_labels, test_data, test_labels = prepare_datasets(samples)
    classifier = instantiate_model()
    classifier.fit(training_data, training_labels, batch_size=batch, epochs=epoch, verbose=1)
    score = classifier.evaluate(test_data, test_labels, batch_size=batch)
    print('OVERALL SCORE', score)
    classifier.save('model/keras_alphanumeric.mod')
