from keras.models import Model
from keras.layers import Dense, Input, Reshape, Conv2D, Activation
from keras.applications.mobilenet import MobileNet, preprocess_input
import numpy as np
from PIL import Image
import time
import requests
# from tts_say import say_sync
import os

config_size = 224
config_alpha = 1

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

class TimeRange():
    def __init__(self, time_start, time_end):
        import math
        b, a = math.modf(time_start)
        assert 0 <= a < 23
        assert 0 <= b < 59
        self.start_time = time_start
        b, a = math.modf(time_end)
        assert 0 <= a < 23
        assert 0 <= b < 59
        assert time_start <= time_end
        self.end_time = time_end

    def __contains__(self, tm):
        if isinstance(tm, float):
            return self.start_time <= tm < self.end_time
        elif isinstance(tm, time.struct_time):
            t = tm.tm_hour + tm.tm_min/100.0
            return self.start_time <= t < self.end_time
        else:
            raise Exception('type error')

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
    # 早晨设置一次
    set_flag_time = TimeRange(5.59, 6.00)
    # 检测区间，只在这段时间的第一次检测到人时才播放简报
    alert_time = TimeRange(6.10, 10.00)
    report_flag = False
    last_img = Image.open(requests.get("http://192.168.1.10/tmpfs/auto.jpg?usr=admin&pwd=admin", stream=True).raw)
    last_img = getVector(last_img)
    i = 0
    model = load_model()
    while True:
        tm = time.localtime()
        if tm in set_flag_time:
            print('Set Flag')
            report_flag = True
        picture = Image.open(requests.get("http://192.168.1.10/tmpfs/auto.jpg?usr=admin&pwd=admin", stream=True).raw)
        current_img = getVector(picture)
        if se(last_img, current_img) < 1.0:
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
        # print('猫:%.2f 狗:%.2f 人:%.2f.jpg' % (pred[0, 0], pred[0, 1], pred[0, 2]))
        if pred[0, 2] > 0.25:
            # img.save('./images/%02d 猫:%.2f 狗:%.2f 人:%.2f.jpg' % (i, pred[0, 0], pred[0, 1], pred[0, 2]))
            # say_sync('你好！')
            if report_flag and (tm in alert_time):
                report_flag = False
                os.system("python3 brief_report.py")

        time.sleep(1)

if __name__ == '__main__':
    main()