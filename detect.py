from keras.models import Model
from keras.layers import Dense, Input, Reshape, Conv2D, Activation
from keras.applications.mobilenet import MobileNet, preprocess_input
import numpy as np
from PIL import Image
import time
import requests
# from tts_say import say_sync
import os

from wechat_bot import send_image, send_text
from sysinfo import get_mem, get_disk, get_cpu_temp, get_acpi_temp

config_size = 224
config_alpha = 1

next_send_report = 0
report_offset = 18* 60 * 60 - 8 * 60 * 60 # 18:00:00

def get_next_time():
    now = int(time.time())
    now = int(now / 86400)
    now += 1
    now *= 86400
    now += report_offset
    return now

next_send_report = 0

def load_model():
    input_layer = Input(shape=(config_size, config_size, 3))
    cnn_model = MobileNet(input_shape=(config_size, config_size, 3), input_tensor=input_layer, alpha=config_alpha, weights=None, include_top=False, pooling='avg')
    x = Reshape((1, 1, int(1024*config_alpha)), name='reshape_1')(cnn_model.output)
    x = Conv2D(3, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('sigmoid', name='activation_preds')(x)
    output_layer = Reshape((3,), name='resharp_2')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    model.load_weights('weigth.dump')
    return model

def getVector(img):
    img = img.convert("L")
    img = img.resize((32, 32), Image.ANTIALIAS)
    img = np.reshape(img, (32*32))
    img = np.asarray(img/255., dtype=float)
    return img

def se(vec1, vec2):
    vec1 -= vec2
    return np.sum(np.power(vec1, [2]))

def main():
    global next_send_report
    last_img = Image.open(requests.get("http://192.168.18.251/tmpfs/auto.jpg?usr=admin&pwd=admin", stream=True).raw)
    last_img = getVector(last_img)
    i = 0
    model = load_model()
    while True:
        picture = Image.open(requests.get("http://192.168.18.251/tmpfs/auto.jpg?usr=admin&pwd=admin", stream=True).raw)
        current_img = getVector(picture)

        if time.time() > next_send_report:
            next_send_report = get_next_time()
            print('next send time %d' % next_send_report)
            send_image(picture)
            info = ''
            info += get_mem()
            info += '\n'
            info += get_disk()
            info += '\n'
            info += get_cpu_temp()
            info += '\n'
            info += get_acpi_temp()
            send_text(info)

        if se(last_img, current_img) < 3.0:
            time.sleep(1)
            continue
        last_img = current_img
        img = picture.resize((224, 224), Image.ANTIALIAS)
        i += 1
        img_new = np.zeros((1, 224, 224, 3), dtype=np.float)
        img_new[0,:,:,:] = np.reshape(np.asarray(img, dtype=np.float), (224, 224, 3))
        preprocess_input(img_new)

        start_time = time.time()
        pred = model.predict(img_new)
        end_time = time.time()
        # print("%.2f" % (end_time - start_time), end='s ')
        # print('cat:%.2f dog:%.2f human:%.2f' % (pred[0, 0], pred[0, 1], pred[0, 2]))
        if pred[0, 2] > 0.25:
            send_image(picture)
            send_text('cat:%.2f dog:%.2f human:%.2f' % (pred[0, 0], pred[0, 1], pred[0, 2]))
            time.sleep(10)

        time.sleep(1)

if __name__ == '__main__':
    main()
