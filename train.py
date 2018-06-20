from keras.models import Model
from keras.layers import Dense, Input, Reshape, Conv2D, Activation, Dropout
from keras.optimizers import SGD, Adadelta, RMSprop
from keras.applications.mobilenet import MobileNet, preprocess_input
import numpy as np
import pickle
import random
from PIL import Image, ImageEnhance


config = {
    'size_x': 224,
    'size_y': 224,
    'alpha': 1,
    'optimizer': RMSprop(lr=0.0001) # SGD(lr=0.0001, momentum=0.9)
}

config_size = 224
config_alpha = 1
config_dropout = 0.25

def build_model():
    input_layer = Input(shape=(config_size, config_size, 3))
    cnn_model = MobileNet(input_shape=(config_size, config_size, 3), input_tensor=input_layer, alpha=config_alpha, include_top=False, pooling='avg')
    # for layer in cnn_model.layers:    # 先把所有的层锁住，用很小的学习率训练一轮
    for layer in cnn_model.layers[:-25]:    # 放开后面若干几层，继续一轮，小学习率
        layer.trainable = False
    x = Reshape((1, 1, int(1024*config_alpha)), name='reshape_1')(cnn_model.output)
    x = Dropout(config_dropout)(x)
    x = Conv2D(3, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('sigmoid', name='activation_preds')(x)
    output_layer = Reshape((3,), name='resharp_2')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse',
                    optimizer=config['optimizer'],
                    metrics=['accuracy'])
    return model

tags = pickle.load(open('tag.pkl', 'rb'))

def img_keep(img):
    return img

def img_zoom_7_5(img):
    img_new = Image.new("RGB", (config_size, config_size))
    # 反正CNN是权值共享的，直接贴到0, 0的位置
    img_new.paste(img.resize((int(config_size*3/4), int(config_size*3/4))), (0, 0))
    img = None
    return img_new

def img_zoom_5_0(img):
    img_new = Image.new("RGB", (config_size, config_size))
    img_new.paste(img.resize((int(config_size/2), int(config_size/2))), (0, 0))
    img = None
    return img_new

image_resize = [img_keep, img_keep, img_keep, img_zoom_7_5, img_zoom_5_0]

def img_enhance_clr(img):
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(random.random())

def img_enhance_cts(img):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(random.uniform(0.7, 1.0))

def img_enhance_brt(img):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(random.uniform(0.3, 1.0))

def img_enhance(img):
    if random.randint(0, 4) == 0:
        img = img_enhance_clr(img)
    if random.randint(0, 4) == 0:
        img = img_enhance_cts(img)
    if random.randint(0, 3) == 0:
        img = img_enhance_brt(img)

    return img

def data_generator():
    import threading
    import queue
    file_Q = queue.Queue(maxsize=4096)
    Q = queue.Queue(maxsize=256)

    class Provider(threading.Thread):
        def run(self):
            while True:
                # 需要的话这里要加入随机排序
                for k, _ in tags.items():
                    file_Q.put(k)

    class Worker(threading.Thread):
        def run(self):
            while True:
                k = file_Q.get()
                file_name = './VOC2012/JPEGImages/%s.jpg' % k
                img = Image.open(file_name)
                img = img.resize((config_size, config_size), Image.ANTIALIAS)
                img = random.choice(image_resize)(img)
                img = img_enhance(img)
                temp = np.asarray(img, dtype=np.float)
                np.expand_dims(temp, -1)
                Q.put((tags[k], temp))

    provider = Provider()
    provider.setDaemon(True)
    worker1, worker2, worker3, worker4 = Worker(), Worker(), Worker(), Worker()
    worker1.setDaemon(True)
    worker2.setDaemon(True)
    worker3.setDaemon(True)
    worker4.setDaemon(True)

    provider.start()
    worker1.start()
    worker2.start()
    worker3.start()
    worker4.start()

    while True:
        tag = np.zeros(shape=(16, 3), dtype=float)
        im = np.zeros(shape=(16, config_size, config_size, 3), dtype=float)
        for i in range(16):
            a, b = Q.get()
            tag[i,:] = a
            im[i,:,:,:] = b
        preprocess_input(im)
        yield (im, tag)

model = build_model()

print(model.summary())

train = False
import os
if os.path.exists('weigth.dump'):
    if_load = input('\n加载权重?(Y/N) ')
    if_load = if_load in ['y', 'Y']
    if if_load:
        model.load_weights('weigth.dump')
        if_train = input('\n继续训练?(Y/N) ') in ['y', 'Y']
        if if_train:
            train = True
    else:
        train = True
else:
    train = True

if train:
    try:
        model.fit_generator(data_generator(), steps_per_epoch=32, epochs=16, verbose=1)
    except KeyboardInterrupt:
        import sys
        sys.exit('Exiting...')
    model.save_weights('weigth.dump')



import random
from PIL import Image

i = 1
for k, v in random.sample(tags.items(), 20):
    file_name = './VOC2012/JPEGImages/%s.jpg' % k
    img_r = Image.open(file_name)
    img = img_r.resize((config_size, config_size), Image.ANTIALIAS)
    picture = np.zeros((1,config_size,config_size,3), dtype=float)
    picture[0,] = np.asarray(img, dtype=float)
    preprocess_input(picture)
    pred = model.predict(picture)
    pic_name = '%02d 猫:%.2f 狗:%.2f 人:%.2f.jpg' % (i, pred[0, 0], pred[0, 1], pred[0, 2])
    print(pic_name)
    img_r.save('./images/'+pic_name)
    i += 1

print('end.')