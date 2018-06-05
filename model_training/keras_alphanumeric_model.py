import cv2
import random
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


letter_dir = 'base_letters/'
img_size = 32


def randomize_image(image):

    image = image[y1:y2, x1:x2]
    # trim down to desired size
    # assume letter is at center of image
    # add random noise
    # add random artifacts
    return 0


def get_label_filename(string):
    char = string.split('_')[2]
    order = ord(char)
    if 'lower' in string:
        return order - 97
    elif 'upper' in string:
        return order - 39
    elif 'number' in string:
        return order + 4


def generate_datasets(count):
    files = os.listdir(letter_dir)
    secure_random = random.SystemRandom()
    results = []
    for i in range(count):
        file = secure_random.choice(files)
        img = cv2.imread(letter_dir + file)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = randomize_image(img)
        results.append((img, get_label_filename(file)))
    return results


if __name__ == '__main__':
    input_shape = (img_size, img_size, 1)
    data = generate_datasets(count=800000)  # returns giant list of tuples where (sample, label)

    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_data, train_labels, batch_size=32, epochs=10, verbose=1)
    score = model.evaluate(test_data, test_labels, verbose=0)
    print(score)
    model.save('model/keras_mnist.mod')