import os
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import random
import numpy as np
import cv2


base_size = 100
img_size = 32
min_font = 24
max_font = 32
font_dir = 'fonts/'
save_dir = 'sample_letters/'
charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
samples_per_class = 1
wiggle = 5


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


def make_random_image(font_file, c, idx):
    # base overhead stuff
    paper = random.randint(150, 255)
    ink = random.randint(0, 75)
    font_size = random.randint(min_font, max_font)
    underscore = bool(random.getrandbits(1))
    font = ImageFont.truetype(font_dir + font_file, font_size)
    left = random.choice(charset)
    right = random.choice(charset)
    string = left + c + right
    # add text
    canvas = Image.new('RGBA', (base_size, base_size), (paper, paper, paper))
    draw = ImageDraw.Draw(canvas)
    w, h = draw.textsize(text=string, font=font)
    w = round((base_size - w) / 2)
    h = round((base_size - h) / 2)
    draw.text((w, h), string, (ink, ink, ink), font=font)
    ImageDraw.Draw(canvas)
    # add noise and crap
    canvas = np.asarray(canvas)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    canvas = crop_image_random(canvas)
    canvas = add_gaussian_noise(canvas)
    if underscore:
        canvas = add_line(canvas, ink)
    # alternative is to return the image here instead of saving for the sake of creating test/training data
    canvas = Image.fromarray(np.uint8(canvas))
    # save file
    filename = font_file.lower().replace('.ttf', '') + '_%s_%s_%s.png' % (charset.index(c), c, idx)
    print(filename)
    canvas.save(save_dir + filename)


if __name__ == '__main__':
    random.SystemRandom()
    random.seed()
    fonts = os.listdir(font_dir)
    for font in fonts:
        for character in charset:
            for i in range(samples_per_class):
                make_random_image(font, character, i)