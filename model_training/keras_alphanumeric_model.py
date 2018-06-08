from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import os
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import random
import numpy as np
import cv2


letter_dir = 'base_letters/'
img_size = 32
batch = 64
epoch = 50
wiggle = 5
base_size = 100
min_font = 24
max_font = 32
samples_per_class = 32 * 16
input_shape = (img_size, img_size, 1)
font_dir = 'fonts/'
save_dir = 'sample_letters/'
charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
num_classes = len(charset)


def add_gaussian_noise(image):
    gauss = np.random.normal(loc=0, scale=20, size=image.shape)
    gauss = gauss.reshape(image.shape)
    noisy = image + gauss
    temp = noisy > 255
    noisy[temp] = 255
    temp = noisy < 0
    noisy[temp] = 0
    return noisy


def crop_image_random(image):
    x = round(base_size / 2 - img_size / 2)
    y = round(base_size / 2 - img_size / 2)
    y += random.randint(0, wiggle)  # add vertical variance
    return image[y:y + img_size, x:x + img_size]


def add_line(image, color):
    y1 = random.randint(16, 30)
    y2 = y1 + random.randint(-4, 4)
    image = cv2.line(image, (0, y1), (31, y2), color=color, thickness=1)
    return image


def make_random_image(font_file, c):
    paper = random.randint(140, 255)
    ink = random.randint(0, 90)
    font_size = random.randint(min_font, max_font)
    underscore = bool(random.getrandbits(1))
    font = ImageFont.truetype(font_dir + font_file, font_size)
    left = random.choice(charset)
    right = random.choice(charset)
    string = left + c + right
    canvas = Image.new('RGBA', (base_size, base_size), (paper, paper, paper))
    draw = ImageDraw.Draw(canvas)
    w, h = draw.textsize(text=string, font=font)
    w = round((base_size - w) / 2)
    h = round((base_size - h) / 2)
    draw.text((w, h), string, (ink, ink, ink), font=font)
    ImageDraw.Draw(canvas)
    canvas = np.asarray(canvas)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    canvas = crop_image_random(canvas)
    canvas = add_gaussian_noise(canvas)
    if underscore:
        canvas = add_line(canvas, ink)
    return canvas


def generate_samples():
    random.SystemRandom()
    random.seed()
    fonts = os.listdir(font_dir)
    results = []
    for font in fonts:
        for character in charset:
            label = charset.index(character)
            print('GENERATING SAMPLES - %s - %s' % (font, character))
            for i in range(samples_per_class):
                sample = make_random_image(font, character)
                results.append((sample, label))
    return results


def instantiate_model():
    print('COMPILING MODEL')
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def prepare_datasets(samples):
    random.shuffle(samples)
    training_set = samples[:round(len(samples) / 10 * 9)]
    test_set = samples[round(len(samples) / 10 * 9):]
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
    samples = generate_samples()
    print('SAMPLES GENERATED', len(samples))
    training_data, training_labels, test_data, test_labels = prepare_datasets(samples)
    classifier = instantiate_model()
    classifier.fit(training_data, training_labels, batch_size=batch, epochs=epoch, verbose=1)
    score = classifier.evaluate(test_data, test_labels, batch_size=batch)
    print('OVERALL SCORE', score)
    classifier.save('model/keras_alphanumeric.mod')
