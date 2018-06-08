import os
import random
import numpy as np
import cv2


img_size = 32
letter_dir = 'base_letters/'
save_dir = 'sample_letters/'
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
    image = cv2.resize(image, dsize=(size, size), fx=factor, fy=factor)
    return image


def randomize_image(image):
    image = add_gaussian_noise(image)
    image = random_brightness(image)
    image = random_size(image)
    image = crop_image_random(image)
    # TODO add underscore to some samples
    return image


if __name__ == '__main__':
    files = os.listdir(letter_dir)
    for f in files:
        print(f)
        img = cv2.imread(letter_dir + f)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = randomize_image(img)
        cv2.imwrite(save_dir + f, img)