# resize pokeGAN.py
import os
import cv2

src = "./data" #original images
dst = "./resizedData" # resized images

os.mkdir(dst)

for i in os.listdir(src):
    image = cv2.imread(os.path.join(src,i))
    image2 = cv2.resize(image,(256,256))
    cv2.imwrite(os.path.join(dst,each), image2)
    