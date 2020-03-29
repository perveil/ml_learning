from PIL import Image

im = Image.open("./1.jpg")
im.rotate(45).show()