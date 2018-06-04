import os
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

fonts = os.listdir('fonts/')
font_size = 20
img_size = 32

# update character lists here to create new data sets for training
lowercase = 'abcdefghijklmnopqrstuvwxyz'
uppercase = lowercase.upper()
numbers = '1234567890'


def save_characters(font, charlist, label):
    for c in charlist:
        img = Image.new("RGBA", (img_size, img_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), c, (0, 0, 0), font=font)
        ImageDraw.Draw(img)
        outfile = filename.lower().replace('.ttf', '') + '_%s_%s.png' % (label, c)
        print('SAVED', outfile)
        img.save('base_letters/' + outfile)


for filename in fonts:
    font = ImageFont.truetype("fonts/%s" % filename, font_size)
    save_characters(font, lowercase, 'lower')
    save_characters(font, uppercase, 'upper')
    save_characters(font, numbers, 'numbers')
