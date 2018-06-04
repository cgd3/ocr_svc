from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

font = ImageFont.truetype("fonts/arial.ttf", 32)
img = Image.new("RGBA", (32, 32), (255, 255, 255))
draw = ImageDraw.Draw(img)
draw.text((0, 0), "Z", (0, 0, 0), font=font)
draw = ImageDraw.Draw(img)
img.save("a_test.png")
