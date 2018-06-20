#!/usr/bin/python3

from PIL import Image
from VideoCapture import Device
import numpy as np
from time import sleep

cap = Device()

def getImage():
    img_buf = cap.getBuffer()
    img = Image.frombytes("RGB", (img_buf[1], img_buf[2]), img_buf[0], "raw", "BGR")
    img = img.convert("L")
    img = img.resize((32, 32), Image.ANTIALIAS)
    img = np.reshape(img, (32*32))
    img = np.asarray(img/255., dtype=float)

    return img

def main():
    last_img = np.zeros((32*32), dtype=float)
    while True:
        current_img = getImage()
        # 求方差
        last_img -= current_img
        last_img = np.power(last_img, [2])
        print("%.2f" % np.sum(last_img))
        last_img = current_img
        sleep(.3)

if __name__ == '__main__':
    main()
