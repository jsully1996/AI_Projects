from PIL import Image
import os
src = "./resizedData"
dst = "./resized/"

for i in os.listdir(src):
    img = Image.open(os.path.join(src,i))
    if img.mode == 'RGBA':
        img.load() # required for img.split()
        background = Image.new("RGB", img.size, (0,0,0))
        background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
        background.save(os.path.join(dst,i.split('.')[0] + '.jpg'), 'JPEG')
    else:
        img.convert('RGB')
        img.save(os.path.join(dst,i.split('.')[0] + '.jpg'), 'JPEG')
