# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import keras
import numpy as np
from keras.models import load_model
from tricks import *
import zipfile
import os

class Paint_MODEL():
    def __init__(self, opts):
        self.session = keras.backend.get_session()
        device_A = '/gpu:0'
        device_B = '/gpu:0'
        
        if not os.path.isdir('weights'):
            print('extract weights files !!')
            with zipfile.ZipFile(opts['models'],"r") as zip_ref:
                zip_ref.extractall(".")

        with tf.device(device_A):

            self.ipa = tf.placeholder(dtype=tf.float32, shape=(None, 1))
            self.ip1 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 1))
            self.ip3 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))
            self.ip4 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 4))
            self.ip3x = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))

            baby = load_model('weights/baby.net')
            baby_place = tf.concat([- 512 * tf.ones_like(self.ip4[:, :, :, 3:4]), 128 * tf.ones_like(self.ip4[:, :, :, 3:4]), 128 * tf.ones_like(self.ip4[:, :, :, 3:4])], axis=3)
            baby_yuv = self.RGB2YUV(self.ip4[:, :, :, 0:3])
            baby_alpha = tf.where(x=tf.zeros_like(self.ip4[:, :, :, 3:4]), y=tf.ones_like(self.ip4[:, :, :, 3:4]), condition=tf.less(self.ip4[:, :, :, 3:4], 128))
            baby_hint = baby_alpha * baby_yuv + (1 - baby_alpha) * baby_place
            self.baby_op = self.YUV2RGB(baby(tf.concat([self.ip1, baby_hint], axis=3)))

            girder = load_model('weights/girder.net')
            self.gird_op = (1 - girder([1 - self.ip1 / 255.0, self.ip4, 1 - self.ip3 / 255.0])) * 255.0

            reader = load_model('weights/reader.net')
            features = reader(self.ip3 / 255.0)
            featuresx = reader(self.ip3x / 255.0)

            head = load_model('weights/head.net')
            feed = [1 - self.ip1 / 255.0, (self.ip4[:, :, :, 0:3] / 127.5 - 1) * self.ip4[:, :, :, 3:4] / 255.0]
            for _ in range(len(features)):
                item = keras.backend.mean(features[_], axis=[1, 2])
                itemx = keras.backend.mean(featuresx[_], axis=[1, 2])
                feed.append(item * self.ipa + itemx * (1 - self.ipa))
            _, _, head_temp = head(feed)

            neck = load_model('weights/neck.net')
            _, _, neck_temp = neck(feed)
            feed[0] = tf.clip_by_value(1 - tf.image.resize_bilinear(self.ToGray(self.VGG2RGB(head_temp) / 255.0), tf.shape(self.ip1)[1:3]), 0.0, 1.0)
            _, _, head_temp = neck(feed)
            self.head_op = self.VGG2RGB(head_temp)
            self.neck_op = self.VGG2RGB(neck_temp)


        with tf.device(device_B):

            self.ip3B = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))

            tail = load_model('weights/tail.net')
            pads = 7
            self.tail_op = tail(tf.pad(self.ip3B / 255.0, [[0, 0], [pads, pads], [pads, pads], [0, 0]], 'REFLECT'))[:, pads*2:-pads*2, pads*2:-pads*2, :] * 255.0


        self.session.run(tf.global_variables_initializer())


        tail.load_weights('weights/tail.net')
        baby.load_weights('weights/baby.net')
        head.load_weights('weights/head.net')
        neck.load_weights('weights/neck.net')
        girder.load_weights('weights/girder.net')
        reader.load_weights('weights/reader.net')

    
    def paint(self, sketch, points = [], reference = None, alpha = 0.5):
        sketch = self.ToGray(np.array(sketch))
        sketch_1024 = k_resize(sketch, 64)

        sketch_256 = mini_norm(k_resize(min_k_down(sketch_1024, 2), 16))
        sketch_128 = hard_norm(sk_resize(min_k_down(sketch_1024, 4), 32))
        print('sketch prepared')

        baby = self.go_baby(sketch_128, opreate_normal_hint(ini_hint(sketch_128), points, type=0, length=1))
        baby = de_line(baby, sketch_128)

        for _ in range(16):
            baby = blur_line(baby, sketch_128)
        baby = self.go_tail(baby)
        baby = clip_15(baby)

        print('baby born')
        composition = self.go_gird(sketch=sketch_256, latent=d_resize(baby, sketch_256.shape), hint=ini_hint(sketch_256))

        composition = self.go_tail(composition)

        painting_function = self.go_head

        result = painting_function(
            sketch=sketch_1024,
            global_hint=k_resize(composition, 14),
            local_hint=opreate_normal_hint(ini_hint(sketch_1024), points, type=2, length=2),
            global_hint_x=k_resize(reference, 14) if reference is not None else k_resize(composition, 14),
            alpha=(1 - alpha) if reference is not None else 1
        )
        result = self.go_tail(result)
        return result[:,:,::-1]
    
    def ToGray(self, x):
        R = x[:, :, :, 0:1]
        G = x[:, :, :, 1:2]
        B = x[:, :, :, 2:3]
        return 0.30 * R + 0.59 * G + 0.11 * B


    def RGB2YUV(self, x):
        R = x[:, :, :, 0:1]
        G = x[:, :, :, 1:2]
        B = x[:, :, :, 2:3]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = 0.492 * (B - Y) + 128
        V = 0.877 * (R - Y) + 128
        return tf.concat([Y, U, V], axis=3)


    def YUV2RGB(self, x):
        Y = x[:, :, :, 0:1]
        U = x[:, :, :, 1:2]
        V = x[:, :, :, 2:3]
        R = Y + 1.140 * (V - 128)
        G = Y - 0.394 * (U - 128) - 0.581 * (V - 128)
        B = Y + 2.032 * (U - 128)
        return tf.concat([R, G, B], axis=3)


    def VGG2RGB(self, x):
        return (x + [103.939, 116.779, 123.68])[:, :, :, ::-1]


    def go_head(self, sketch, global_hint, local_hint, global_hint_x, alpha):
        return self.session.run(self.head_op, feed_dict={
            self.ip1: sketch[None, :, :, None], self.ip3: global_hint[None, :, :, :], self.ip4: local_hint[None, :, :, :], self.ip3x: global_hint_x[None, :, :, :], self.ipa: np.array([alpha])[None, :]
        })[0].clip(0, 255).astype(np.uint8)


    def go_neck(self, sketch, global_hint, local_hint, global_hint_x, alpha):
        return self.session.run(self.neck_op, feed_dict={
            self.ip1: sketch[None, :, :, None], self.ip3: global_hint[None, :, :, :], self.ip4: local_hint[None, :, :, :], self.ip3x: global_hint_x[None, :, :, :], self.ipa: np.array([alpha])[None, :]
        })[0].clip(0, 255).astype(np.uint8)


    def go_gird(self, sketch, latent, hint):
        return self.session.run(self.gird_op, feed_dict={
            self.ip1: sketch[None, :, :, None], self.ip3: latent[None, :, :, :], self.ip4: hint[None, :, :, :]
        })[0].clip(0, 255).astype(np.uint8)


    def go_tail(self, x):
        return self.session.run(self.tail_op, feed_dict={
            self.ip3B: x[None, :, :, :]
        })[0].clip(0, 255).astype(np.uint8)


    def go_baby(self, sketch, local_hint):
        return self.session.run(self.baby_op, feed_dict={
            self.ip1: sketch[None, :, :, None], self.ip4: local_hint[None, :, :, :]
        })[0].clip(0, 255).astype(np.uint8)


