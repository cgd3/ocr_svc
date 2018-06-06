import os
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

fonts = os.listdir('fonts/')
font_size = 24
img_size = 100  # image will be square
charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def save_characters(font, charlist):
    for c in charlist:
        img = Image.new("RGBA", (img_size, img_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        w, h = draw.textsize(text=c, font=font)
        w = round((img_size - w) / 2)
        h = round((img_size - h) / 2)
        draw.text((w, h), c, (0, 0, 0), font=font)
        ImageDraw.Draw(img)
        outfile = filename.lower().replace('.ttf', '') + '_%s_%s.png' % (charlist.index(c), c)
        print('SAVED', outfile)
        img.save('base_letters/' + outfile)


for filename in fonts:
    font = ImageFont.truetype("fonts/%s" % filename, font_size)
    save_characters(font, charset)
